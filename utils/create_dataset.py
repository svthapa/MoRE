import torch
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import transforms, RandAugment
import numpy as np
from scipy.signal import resample_poly
from PIL import Image
from skimage import exposure
import wfdb
import lmdb
import pickle 
from collections import deque 
from transformers import RobertaTokenizerFast, AutoTokenizer
# Baseline wander removal
import scipy
import scipy.signal
import h5py
# from randaugment import RandomAugment

CHEXPERT_UNCERTAIN_MAPPINGS = {
    "Atelectasis": 1,
    "Cardiomegaly": 0,
    "Consolidation": 0,
    "Edema": 1,
    "Pleural Effusion": 1,
}

label_index_mapping = {
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Consolidation": 2,
    "Edema": 3,
    "Enlarged Cardiomediastinum": 4,
    "Fracture": 5,
    "Lung Lesion": 6,
    "Lung Opacity": 7,
    "No Finding": 8,
    "Pleural Effusion": 9,
    "Pleural Other": 10,
    "Pneumonia": 11,
    "Pneumothorax": 12,
    "Support Devices": 13
}

def replace_uncertain_labels(item, label_index_mapping, uncertain_mappings):
    """
    Replace -1 values in the binary labels array for specific conditions.

    :param item: List where the 4th item is an array of binary labels.
    :param label_index_mapping: Dictionary mapping condition names to label indices.
    :param uncertain_mappings: Dictionary defining replacement values for -1 in uncertain cases.
    :return: Modified item with replaced uncertain labels.
    """
    # Extract the labels array from the item
    labels_array = item.copy()

    # Iterate over the uncertain_mappings to find and replace -1 values
    for condition, replacement_value in uncertain_mappings.items():
        if condition in label_index_mapping:  # Ensure the condition is in the index mapping
            index = label_index_mapping[condition]  # Get the index of the condition
            if labels_array[index] == -1.0:  # Check if the label is -1 (uncertain)
                labels_array[index] = replacement_value  # Replace with the specified value

    # Update the item with the modified labels array
    # item = labels_array
    return labels_array

def baseline_wander_removal(data, sampling_frequency = 100):
    row,__ = data.shape
    processed_data = np.zeros(data.shape)
    for lead in range(0,row):
        # Baseline estimation
        win_size = int(np.round(0.2 * sampling_frequency)) + 1
        baseline = scipy.signal.medfilt(data[lead,:], win_size)
        win_size = int(np.round(0.6 * sampling_frequency)) + 1
        baseline = scipy.signal.medfilt(baseline, win_size)
        # Removing baseline
        filt_data = data[lead,:] - baseline
        processed_data[lead,:] = filt_data
    return processed_data

class MultiModalDataset(Dataset):
    def __init__(self, paths, ecg_augmentor, phase = 'train'):
        self.paths = paths
        self.xray_transforms = transforms.Compose([
                    # transforms.AutoAugment(),
                    # transforms.RandomHorizontalFlip(p=0.5),
                    # RandAugment(num_ops = 2, magnitude= 9),
                    transforms.RandomApply([transforms.RandomResizedCrop(224, scale=(0.3, 0.9))], p = 0.8),
                    transforms.RandomApply([transforms.ColorJitter(brightness =0.3, contrast =0.3)], p=0.8),
                    transforms.RandomApply([transforms.GaussianBlur(kernel_size=7)], p=0.5),
                    transforms.RandomHorizontalFlip(p = 0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.499], [0.293])
        ])
        self.val_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.499], [0.293])
        ])
        self.augmentor = ecg_augmentor
        self.phase = phase 
        # self.subjects = self._group_by_subject()
     
    def _group_by_subject(self):
        subjects = {}
        for xray, ecg, _, _ in self.paths:
            # Extracting the subject ID from the path
            subject_id = xray.split("/p")[2].split("/")[0]
            if subject_id not in subjects:
                subjects[subject_id] = []
            subjects[subject_id].append((xray, ecg))
        return subjects
    
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        xray_path = self.paths[idx][0]
        # subject_id = xray_path.split('/')[-3][1:]
        ecg_path = self.paths[idx][1]
        label = self.paths[idx][2]
        label_tensor = self._preprocess_label(label)
        xray=self._preprocess_xray(xray_path)
        ecg = self._preprocess_ecg(ecg_path)
        return xray, ecg, label_tensor
    
    def _preprocess_label(self,label):
        # label = replace_uncertain_labels(label, label_index_mapping, CHEXPERT_UNCERTAIN_MAPPINGS)
        label = label[[0,1,3,9]].astype(float)
        # label_tensor = torch.tensor(label, dtype=torch.float32)
        return label
        
    def _preprocess_xray(self, x):
        x = Image.open(x)
        x = x.convert('L')
        x = x.resize((224,224))
        x = np.array(exposure.equalize_adapthist(x/np.max(x)))
        x = x *255 
        x = x.astype(np.uint8)
        x = Image.fromarray(x)
        if self.phase == 'train':
            x1 = self.xray_transforms(x)
        else:
            x1 = self.val_transforms(x)
        
        return torch.cat([x1,x1,x1], dim = 0)

    
    def _preprocess_ecg(self,x):
        x =self._load_ecg(x)
        x = self._remove_nan(x)
        x = resample_poly(x, up = 1, down = 5, axis = 0)
        x = np.transpose(x, (1,0))
        x = baseline_wander_removal(x, 100)
        x = self.normalize_per_lead(x)
        if self.phase == 'train':
            # x = self.augmentor.augment(x)
            x = torch.from_numpy(x).float()
        else:
            x = torch.from_numpy(x).float()
        
        return x
        
    def _load_ecg(self, x):
        record = wfdb.rdsamp(x)
        return record[0]
    
    def _remove_nan(self,x):
        nan_indices = np.argwhere(np.isnan(x))
        x[np.isnan(x)] = 0
        return x 
    
    def _normalize_ecg(self, ecg):
        smoothed_waveform = np.empty_like(ecg)
        for i in range(ecg.shape[0]):
            smoothed_waveform[i, :] = savgol_filter(ecg[i, :], window_length=5, polyorder=2)
        return smoothed_waveform
    
    def _normalize(self,ecg_data):
        min_val = np.min(ecg_data)
        max_val = np.max(ecg_data)

        epsilon = 1e-10
        normalized_data = (2 * (ecg_data - min_val) / (max_val - min_val + epsilon)) - 1
        return normalized_data
    
    def normalize_per_lead(self,ecg_data):
        normalized_data = np.zeros_like(ecg_data, dtype=float)

        # Iterate over each lead
        for i in range(ecg_data.shape[0]):
            lead_data = ecg_data[i, :]
            min_val = np.min(lead_data)
            max_val = np.max(lead_data)

            epsilon = 1e-10
            normalized_lead = (2 * (lead_data - min_val) / (max_val - min_val + epsilon)) - 1

            # Assign normalized lead data to the corresponding row in the output array
            normalized_data[i, :] = normalized_lead

        return normalized_data
    
    
class MultiModalLMDBDataset(Dataset):
    def __init__(self, lmdb_path, ecg_augmentor, phase = 'train', indices = None):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, max_readers=4096, meminit=False)
        with self.env.begin(write=False) as txn:
            self.total_length = txn.stat()['entries']
            self.indices = indices if indices is not None else np.arange(self.total_length)
        
        
        self.xray_transforms = transforms.Compose([
                    # transforms.AutoAugment(),
                    # transforms.RandomHorizontalFlip(p=0.5),
                    # RandAugment(num_ops = 2, magnitude= 9),
                    transforms.RandomApply([transforms.RandomResizedCrop(224, scale=(0.5, 0.9),interpolation=Image.BICUBIC)], p = 0.8),
                    transforms.RandomApply([transforms.ColorJitter(brightness =0.2, contrast =0.2)], p=0.8),
                    transforms.RandomApply([transforms.GaussianBlur(kernel_size=7)], p=0.5),
                    transforms.RandomHorizontalFlip(p = 0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.499], [0.293])
        ])
        self.val_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.499], [0.293])
        ])
        
        self.augmentor = ecg_augmentor
        self.phase = phase
    
        

    def __len__(self):
        return len(self.indices)

    def _deserialize_data(self, serialized_data):
        data = pickle.loads(serialized_data)
        
        # Convert X-ray bytes back to numpy array
        xray_data = np.frombuffer(data["xray"], dtype=np.uint8).reshape(224,224) 
        
        # Convert ECG bytes back to numpy array
        ecg_data = np.frombuffer(data["ecg"]).reshape(12, 1000)  # Adjust dtype if needed
        
        labels = np.frombuffer(data["label"], dtype='float64')
        # print(type(data['label']))
        
        return xray_data, ecg_data, labels

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        with self.env.begin(write=False) as txn:
            serialized_data = txn.get(str(actual_idx).encode('utf-8'))
        
        xray_data, ecg_data, labels = self._deserialize_data(serialized_data)
        
        xray_data_copy= np.copy(xray_data)
        ecg_data_copy = np.copy(ecg_data)
        
        ecg_data_copy = baseline_wander_removal(ecg_data_copy)
        # labels = replace_uncertain_labels(labels, label_index_mapping, CHEXPERT_UNCERTAIN_MAPPINGS)
        labels = labels[[0,1,3,9]]
        
        # xray_data_copy = np.stack((xray_data_copy,) * 3, axis=-1)
        xray = Image.fromarray(xray_data_copy)
        # Data transformations and augmentations
        if self.phase == 'train':
            xray = self.xray_transforms(xray)
            # ecg = self.augmentor.augment(ecg_data_copy)
            ecg = torch.from_numpy(ecg_data_copy).float()
        else:
            xray = self.val_transforms(xray)
            ecg = torch.from_numpy(ecg_data_copy).float()
        
        
        return (
            torch.cat([xray, xray, xray], dim=0),
            # xray,
            ecg, 
            labels
        )
    
    
class EfficientSubjectBasedSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.subject_ids = list(self.dataset.subjects.keys())
        self.counters = {subj_id: 0 for subj_id in self.subject_ids}
        self.path_to_index = {(str(path_pair[0]), str(path_pair[1])): idx for idx, path_pair in enumerate(self.dataset.paths)}
        
    def __iter__(self):
        np.random.shuffle(self.subject_ids)
        indices = []
        subject_queue = deque(self.subject_ids)  # Using deque for efficient popping from front

        while len(indices) < len(self.dataset):
            if not subject_queue:  # Refill only when queue is empty
                subject_queue.extend(self.subject_ids)
                np.random.shuffle(subject_queue)

            current_subjects = [subject_queue.popleft() for _ in range(min(self.batch_size, len(subject_queue)))]

            for subj in current_subjects:
                current_idx_set = self.dataset.subjects[subj]
                if self.counters[subj] < len(current_idx_set):
                    indices.append(self._find_original_index(current_idx_set[self.counters[subj]]))
                    self.counters[subj] = (self.counters[subj] + 1) % len(current_idx_set)

        return iter(indices)

    def _find_original_index(self, path_pair):
        return self.path_to_index[(str(path_pair[0]), str(path_pair[1]))]

    def __len__(self):
        return len(self.dataset)
    
    
class EcgNotesDataset(Dataset):
    def __init__(self, data, ecg_augmentor, phase = 'train'):
        self.data = data
        self.augmentor = ecg_augmentor
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.phase = phase
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ecg_path = self.data[idx][0]
        note = self.data[idx][1]
        
        ecg = self._preprocess_ecg(ecg_path)
        note = self.encode_note(note)
        
        return ecg, note['input_ids'].squeeze(0), note['attention_mask'].squeeze(0)
        
    def _preprocess_ecg(self,x):
        x =self._load_ecg(x)
        x = self._remove_nan(x)
        x = resample_poly(x, up = 1, down = 5, axis = 0)
        x = np.transpose(x, (1,0))
        x = self._normalize(x)
        if self.phase == 'train':
            x = self.augmentor.augment(x)
            x = torch.from_numpy(x).float()
        else:
            x = torch.from_numpy(x).float()
        
        return x
        
    def _load_ecg(self, x):
        record = wfdb.rdsamp(x)
        return record[0]
    
    def _remove_nan(self,x):
        nan_indices = np.argwhere(np.isnan(x))
        x[np.isnan(x)] = 0
        return x 
    
    def _normalize(self,ecg_data):
        min_val = np.min(ecg_data)
        max_val = np.max(ecg_data)

        epsilon = 1e-10
        normalized_data = (2 * (ecg_data - min_val) / (max_val - min_val + epsilon)) - 1
        return normalized_data
    
    def encode_note(self, x):
        return self.tokenizer(x, return_tensors="pt", padding="max_length", max_length=512, \
                              truncation=True, return_attention_mask=True)
    
    
class MultiModalData(Dataset):
    def __init__(self, data, ecg_augmentor, phase = 'train'):
        self.data = data
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        # special_tokens_dict = {'additional_special_tokens': ['<xray>', '</xray>', '<ecg>', '</ecg>']}
        # self.tokenizer.add_special_tokens(special_tokens_dict)
        # self.tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-7b")
        # self.tokenizer.padding_side = 'right'
        
        self.xray_transforms = transforms.Compose([
                    transforms.RandomApply([transforms.RandomResizedCrop(224, scale=(0.2, 1.0))], p = 0.8),
                    transforms.RandomApply([transforms.ColorJitter(brightness =0.4, contrast =0.4)], p=0.8),
                    transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=0.5),
                    transforms.RandomHorizontalFlip(p = 0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.499], [0.293])
        ])
        self.val_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.499], [0.293])
        ])
        self.augmentor = ecg_augmentor
        self.phase = phase
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        xray = self.preprocess_xray(self.data[idx][0])
        ecg = self._load_ecg(self.data[idx][1])
        xray_note = self.data[idx][2]
        ecg_note = self.data[idx][3]
        note = self.encode_note(xray_note, ecg_note)
        # label = self.data[idx][4][[1,3,9]].astype(float)
        return xray, ecg, note['input_ids'].squeeze(0), note['attention_mask'].squeeze(0)
        
    def preprocess_xray(self, x):
        x = Image.open(x)
        x = x.convert('L')
        x = x.resize((224,224))
        x = np.array(exposure.equalize_adapthist(x/np.max(x)))
        x = x *255 
        x = x.astype(np.uint8)
        x = Image.fromarray(x)
        if self.phase == 'train':
            x = self.xray_transforms(x)
        else:
            x = self.val_transforms(x)
        
        return torch.cat([x,x,x], dim = 0)
    
    def _load_ecg(self, x):
        x = wfdb.rdsamp(x)[0]
        nan_indices = np.argwhere(np.isnan(x))
        x[np.isnan(x)] = 0
        x = resample_poly(x, up = 1, down = 5, axis = 0)
        x = np.transpose(x, (1,0))
        x = baseline_wander_removal(x, 100)
        x = self.normalize_per_lead(x)
        
        if self.phase == 'train':
            x = self.augmentor.augment(x)
            x = torch.from_numpy(x).float()
        else:
            x = torch.from_numpy(x).float()
            
        return x
    
    def normalize_per_lead(self,ecg_data):
        normalized_data = np.zeros_like(ecg_data, dtype=float)

        # Iterate over each lead
        for i in range(ecg_data.shape[0]):
            lead_data = ecg_data[i, :]
            min_val = np.min(lead_data)
            max_val = np.max(lead_data)

            epsilon = 1e-10
            normalized_lead = (2 * (lead_data - min_val) / (max_val - min_val + epsilon)) - 1

            # Assign normalized lead data to the corresponding row in the output array
            normalized_data[i, :] = normalized_lead

        return normalized_data
    
    def encode_note(self, x, y):
        return self.tokenizer(x,y, return_tensors="pt", add_special_tokens = True, max_length=512, padding="max_length", \
                              truncation=True, return_attention_mask=True)
    

class H5Dataset(Dataset):
    def __init__(self, h5_file_path, phase = 'train', indices=None):
        self.h5_file_path = h5_file_path
        self.indices = indices or []
        # Lazy loading: Open the file in __getitem__
        self.xray_transforms = transforms.Compose([
            transforms.RandomApply([transforms.RandomResizedCrop(224, scale=(0.6, 0.9))], p=0.8),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4)], p=0.8),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=0.5),
            transforms.ToTensor(),
        ])
        self.phase = phase
        # self.preload_data()

    def __len__(self):
        if not self.indices:
            with h5py.File(self.h5_file_path, 'r') as file:
                self.indices = list(file.keys())
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        
        # Use a context manager to ensure the file is opened and closed efficiently
        with h5py.File(self.h5_file_path, 'r') as file:
            group_name = f'data_point_{actual_idx}'
            group = file[group_name]
            xray_data = group['xray_data'][:]
            ecg_data = group['ecg_data'][:]
            text_id = torch.tensor(group['text_input_ids'][:]).squeeze(0)
            mask = torch.tensor(group['text_mask'][:]).squeeze(0)
        
        # xray_data, ecg_data, text_id, mask = self.preloaded_data[actual_idx]
        # Process X-ray data
        xray = self.process_xray(xray_data)

        # Convert ECG data to tensor
        ecg = torch.from_numpy(ecg_data).float()

        return xray, ecg, text_id, mask
    
    def process_xray(self,x):
        x = Image.fromarray(x)
        if self.phase == 'train':
            x = self.xray_transforms(x)
        else:
            x = transforms.ToTensor()(x)
        
        return torch.cat([x,x,x], dim = 0)
            
    def close(self):
        # Close the H5 file if it's open
        self.h5_file.close()
        
    def preload_data(self):
        self.preloaded_data = []

        # Open HDF5 file and load data into memory
        with h5py.File(self.h5_file_path, 'r') as file:
            for idx in self.indices:
                group_name = f'data_point_{idx}'
                group = file[group_name]
                xray_data = group['xray_data'][:]
                ecg_data = group['ecg_data'][:]
                text_id = torch.tensor(group['text_input_ids'][:]).squeeze(0)
                mask = torch.tensor(group['text_mask'][:]).squeeze(0)

                # Append data to preloaded cache
                self.preloaded_data.append((xray_data, ecg_data, text_id, mask))

        
class XrayDataset(Dataset):
    def __init__(self, data, phase = 'train'):
        self.data = data
        self.xray_transforms = transforms.Compose([
            transforms.RandomApply([transforms.RandomResizedCrop(224, scale=(0.2, 1.0))], p=0.8),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.8),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=7)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.499], [0.293])
        ])
        self.phase = phase
        self.val_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.499], [0.293])
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        xray = self.data[idx][0]
        label = self.data[idx][1][[0,1,3,9]].astype(float)
        xray = self.preprocess_xray(xray)
        
        return xray, label
    
    def preprocess_xray(self, x):
        x = Image.open(x)
        x = x.convert('L')
        x = x.resize((224,224))
        x = np.array(exposure.equalize_adapthist(x/np.max(x)))
        x = x *255 
        x = x.astype(np.uint8)
        x = Image.fromarray(x)
        if self.phase == 'train':
            x1 = self.xray_transforms(x)
        else:
            x1 = self.val_transforms(x)
        
        return torch.cat([x1,x1,x1], dim = 0)