import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset
import sys
sys.path.append('/path/to/MoRE/utils')
from build_model import MultiModal


def clean_text(text):
    text.replace('\n', ' ')
    # text.replace('_', ' ')
    text.replace('FINDINGS:', '').replace('IMPRESSION:', '')
    text = re.sub(r'\d+\.', '', text)
    text = re.sub(r'[_\u2013\u2014\u2015\u2212\uFE58\uFE63\uFF0D]', '', text)  # Remove all underscore-like characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip() 


class XrayNotesDataset(Dataset):
    def __init__(self, data, phase = 'train'):
        self.data = data
        self.xray_transforms = transforms.Compose([
                    transforms.RandomApply([transforms.RandomResizedCrop(224, scale=(0.6, 0.9))], p = 0.8),
                    transforms.RandomApply([transforms.ColorJitter(brightness =0.4, contrast =0.4)], p=0.8),
                    transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=0.5),
                    # transforms.RandomHorizontalFlip(p = 0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.499], [0.293])
        ])
        self.val_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.499], [0.293])
        ])
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        # special_tokens_dict = {'additional_special_tokens': ['<xray>', '</xray>', '<ecg>', '</ecg>']}
        # self.tokenizer.add_special_tokens(special_tokens_dict)
        # self.tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-7b")
        # self.tokenizer.padding_side = 'right'
        self.phase = phase
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        xray_path = self.data[idx][0]
        note = self.data[idx][1]
        
        xray = self._preprocess_xray(xray_path)
        note = self.encode_note(note)
        
        return xray, note['input_ids'].squeeze(0), note['attention_mask'].squeeze(0)
        
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
    
    def encode_note(self, x):
        return self.tokenizer(x, return_tensors="pt", padding="max_length", max_length=512, \
                              truncation=True, return_attention_mask=True)
    

def get_feature_embeddings(val_loader, model, feature = 'image'):
    model.eval()

    feature_embeddings = []
    attn_scores = []
    with torch.no_grad():
        for xray, note_id, note_attn_mask in tqdm(val_loader):
            if feature == 'image':
                xray = xray.cuda()
                with autocast():
                    image_features = model.xray_model(xray)
                    image_features = model.projector_xray(image_features)
                    
                feature_embeddings.append(image_features)
            else:
                note_id, note_attn_mask = note_id.cuda(), note_attn_mask.cuda()
                with autocast():
                    out = model.text_model(note_id, attention_mask = note_attn_mask).last_hidden_state.mean(dim=1)
        
                    text_features = model.projector_text(
                        out
                        )
                # attn = out[2]
                    
                feature_embeddings.append(text_features)
                # attn_scores.append(attn)
    return model, torch.cat(feature_embeddings)


def find_matches(model, feature_embeddings, query, data_split, n = 6, retrieval = 'image'):
    
    model.eval()
    if retrieval == 'image':
        tokenizer = val.tokenizer
        encoded_query = val.encode_note([query])
        batch = {
            key: torch.tensor(values).cuda()
            for key, values in encoded_query.items()
        }

        with torch.no_grad():
            with autocast():
                text_features = model.text_model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                ).last_hidden_state.mean(dim=1)

                query_embedding = model.projector_text(text_features)
    else:
        query = val._preprocess_xray(query)
        with torch.no_grad():
            with autocast():
                query_embedding = model.xray_model(query.unsqueeze(0).cuda())
                query_embedding = model.projector_xray(query_embedding)

    feature_embeddings_n = F.normalize(feature_embeddings, dim = -1)
    query_embeddings_n = F.normalize(query_embedding, dim = -1)
    dot_similarity = query_embeddings_n @ feature_embeddings_n.T

    values, indices = torch.topk(dot_similarity.squeeze(0), n*5)
    # print(indices)
    matches = [(data_split[idx][0], data_split[idx][1]) for idx in indices[::5]]

    _, axes = plt.subplots(int(n/2), int(n/2), figsize=(7, 7))
    for i, ((img, text), ax) in enumerate(zip(matches, axes.flatten())):
        image = Image.open(img).convert('L').resize((224,224))
        print(f'Text for image: {i+1}')
        print(text)
        print('-'*40)
        ax.imshow(image)
        ax.axis("off")

    # plt.show()
    plt.savefig('grid_plot_retrieval.png')
    
    
chexpert = pd.read_csv('./data/chexpert_5x200.csv')
chexpert_arr = chexpert.values

val = XrayNotesDataset(chexpert_arr, phase = 'val')

model = MultiModal()
model.load_state_dict(model_dict)

batch_size = 36
val_loader = DataLoader(val, batch_size = batch_size, num_workers = 7, shuffle = False)

model, feature_embeddings= get_feature_embeddings(val_loader, model, 'image')

find_matches(model,
             feature_embeddings,
             query="there is cardiomegaly, no edema, no consolidation, no effusion.",
             # query = chexpert_arr[21][0],
             data_split = chexpert_arr,
             n=6,
            retrieval = 'image')

