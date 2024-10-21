import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
from skimage import exposure

import sys
sys.path.append('./utils')
from build_model import ViTModelXray, ProjectionHead, MultiModal
from metrics import calculate_multilabel_map, plot_precision_recall_curve, plot_roc_curve, calc_roc_auc
from sklearn.metrics.pairwise import cosine_similarity

ptbxl = pd.read_csv('../data/ptbxl.csv')
data = ptbxl[['filename_lr', 'diagnostic_superclass', 'strat_fold']].values
data = [[path, note, fold] for path, note, fold in data if os.path.exists(path+'.hea')]

for i, item in enumerate(data):
    if 'NORM' in item[1]:
        data[i][1] = np.array([1, 0, 0, 0, 0])
    elif 'STTC' in item[1]:
        data[i][1] = np.array([0, 1, 0, 0, 0])
    elif 'MI' in item[1]:
        data[i][1] = np.array([0, 0, 1, 0, 0])
    elif 'HYP' in item[1]:
        data[i][1] = np.array([0, 0, 0, 1, 0])
    elif 'CD' in item[1]:
        data[i][1] = np.array([0, 0, 0, 0, 1])
        
train_split = [
    item for item in data \
    if item[2] != 10 and item[2] != 9
    ]

val_split = [
    item for item in data \
    if item[2] == 9
    ]

test_split = [
    item for item in data \
    if item[2] == 10
    ]

class EcgDataset(Dataset):
    def __init__(self, data, phase = 'train'):
        self.data = data
        self.phase = phase
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ecg = self.data[idx][0]
        label = self.data[idx][1]
        ecg = self._preprocess_ecg(ecg)
        # age = self.data[idx][2]
        # label = self._process_label(label)
        return ecg, label
    
    def _preprocess_ecg(self, ecg):
        ecg = self._load_ecg(ecg)
        ecg = self._remove_nan(ecg)
        ecg = np.transpose(ecg, (1,0))
        ecg = self.baseline_wander_removal(ecg, 100)
        ecg = self.normalize_per_lead(ecg)
        ecg = torch.from_numpy(ecg).float()
        return ecg
    
    def _process_label(self, label):
        label = self.label_idx[label].astype(float)
        return label
    
    def _load_ecg(self, x):
        record = wfdb.rdsamp(x)
        return record[0]
    
    def _remove_nan(self,x):
        nan_indices = np.argwhere(np.isnan(x))
        x[np.isnan(x)] = 0
        return x 
    
    def baseline_wander_removal(self,data, sampling_frequency = 100):
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
    
test = EcgDataset(test_split, 'val')

ecgs = []
ground_truth = []
for item in test:
    ecgs.append(item[0])
    ground_truth.append(item[1])
    
ecgs = torch.stack(ecgs)
ground_truth = np.array(ground_truth)

CLASS_PROMPTS = {
    '0': ['Normal ECG'],
    '1': ['ST T Change'],
    '2': ['Myocardial Infarction'],
    '3': ['Hypertrophy'],
    '4': ['Conduction Disturbance']
    
}

model = MultiModal()

class PreprocessNote:
    def __init__(self):
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        # special_tokens_dict = {'additional_special_tokens': ['<xray>', '</xray>', '<ecg>', '</ecg>']}
        # self.tokenizer.add_special_tokens(special_tokens_dict)
        
    def encode_note(self,x):
        return self.tokenizer(x, return_tensors="pt", padding="max_length", max_length=128, \
                                      truncation=True, return_attention_mask=True)
        
    def encode(self, x):
        return self.encode_note(x)
        
def normalize(similarities, method="norm"):

    if method == "norm":
        return (similarities - similarities.mean(axis=0)) / (similarities.std(axis=0))
    elif method == "standardize":
        return (similarities - similarities.min(axis=0)) / (
            similarities.max(axis=0) - similarities.min(axis=0)
        )
    else:
        raise Exception("normalizing method not implemented")
        
def get_embeddings_similarity(model, ecgs, texts):
    model.cuda()
    model.eval()

    ecgs = ecgs.cuda()


    # Assuming your model and data are small enough to fit in memory, you could directly stack inputs
    text_ids = torch.stack([item['input_ids'].squeeze(0) for item in texts])
    masks = torch.stack([item['attention_mask'].squeeze(0) for item in texts])

    text_ids = text_ids.cuda()
    masks = masks.cuda()
    
    with torch.no_grad():
        ecg_feature = model.projetor_ecg_text(model.ecg_model(ecgs))
        text_feature = model.text_model(text_ids, attention_mask = masks).last_hidden_state.mean(dim=1)
        text_feature = model.projector_text(text_feature)
        

    ecg_embeddings = F.normalize(ecg_feature, dim = -1).cpu().numpy()
    text_embeddings = F.normalize(text_feature, dim = -1).cpu().numpy()
    # dot_similarity = ecg_embeddings @ text_embeddings.T
    cos_sim = cosine_similarity(ecg_embeddings, text_embeddings)
    return torch.tensor(cos_sim, device='cuda')

    # return dot_similarity
    
processed_prompts = {}
process_note = PreprocessNote()
for key, notes in CLASS_PROMPTS.items():
    for note in notes:
        if key in processed_prompts:
            processed_prompts[key].append(process_note.encode(note))
        else:
            processed_prompts[key] = [process_note.encode(note)]
            
class_similarities = []

for key, note in processed_prompts.items():
    similarities = get_embeddings_similarity(model, ecgs, note)
    cls_similarity, _ = torch.max(similarities, dim =1)  # average between class prompts
    # cls_similarity = F.softmax(similarities, dim =0)
    class_similarities.append(cls_similarity.cpu())

class_similarities = np.stack(class_similarities, axis=1)

class_similarities = normalize(class_similarities)

from sklearn.metrics import average_precision_score, roc_auc_score

auc_scores = {}
auprc_scores = {}
for i,label in enumerate(label_index_mapping.keys()):
    auc = roc_auc_score(ground_truth[:,i], class_similarities[:,i])
    auprc = average_precision_score(ground_truth[:,i], class_similarities[:,i])
    auc_scores[label] = auc
    auprc_scores[label] = auprc
    
for label, auc in auc_scores.items():
    print(f"{label}: AUC = {auc:.4f}")
    
for label, auprc in auprc_scores.items():
    print(f"{label}: AUPRC = {auprc:.4f}")