import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
# import wandb
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import average_precision_score
import random

import sys
sys.path.append('./utils')

from metrics import calculate_multilabel_map, plot_precision_recall_curve, plot_roc_curve, calc_roc_auc
from build_model import MultiModalHead
from create_dataset import XrayDataset
from ecg_augmentations import ECGAugmentor
from early_stop import EarlyStopping

mimic_data = pd.read_csv('./data/cxr_metadata_chexpert.csv') #file with mimic iv path, and labels

data = [[item[28], item[14:27].astype(float), item[13]] for item in mimic_data.values]

train_items = [item for index, item in enumerate(data) if item[2] == 'train' and -1 not in item[1]]
val_items = [item for index, item in enumerate(data) if item[2] == 'validate' and -1 not in item[1]]
test_split = [item for item in data if item[2] == 'test' and -1 not in item[1]]


random.seed(42)
# Calculate 10% of the length of the indices list
ten_percent = int(len(train_items) * 1)
# Randomly select 10% of the indices
train_in = random.sample(train_items , ten_percent)

ecg_augmentor = ECGAugmentor()
train = MultiModalDataset(train_in, ecg_augmentor, phase = 'train')
val = MultiModalDataset(val_items, ecg_augmentor, phase = 'val')
test = MultiModalDataset(test_split, ecg_augmentor, phase = 'val')

batch_size = 64
train_loader = DataLoader(train, batch_size = batch_size, num_workers = 14, shuffle = True, pin_memory = True)
val_loader = DataLoader(val, batch_size = batch_size, num_workers = 14, shuffle = False, pin_memory = True)
test_loader = DataLoader(test, batch_size = batch_size, num_workers = 14, shuffle = False, pin_memory = True)

model = MultiModalHead(in_dim = 128*2, out_dim = 4, ecg_drop = 0, projector=False)
model.load_state_dict(model_dict)

class XrayModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.xray_model = model.xray_model
        self.projector_xray = model.projector_xray
        self.fc = nn.Linear(128, 4)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        out = self.projector_xray(self.xray_model(x))
        out = self.dropout(out)
        out = self.fc(out)
        return out 

model_xray = XrayModel(model)

num_layers = 12  # Total number of layers
start_ratio = 0.2  # Starting mask ratio

# Calculate linearly decreasing values for each layer
mask_ratios = [start_ratio * (1 - i / (num_layers - 1)) for i in range(num_layers)]

for i, ratio in enumerate(mask_ratios):
    model_xray.xray_model.vit_model.blocks[i].mask_ratio = ratio
    
layers_xray = list(model_xray.xray_model.vit_model.children())

for name, param in layers_xray[4][9:].named_parameters():
    if 'qkv.weight' in name:# or 'bias' in name:
    # or 'bias' in name:
    # or 'proj.weight' in name or 'proj.bias' in name \
        param.requires_grad=True
         
    
for name, param in layers_ecg[4][9:].named_parameters():
    if 'qkv.weight' in name:# or 'bias' in name:
    # or 'bias' in name:
    # or 'proj.weight' in name or 'proj.bias' in name \
        param.requires_grad=True
    
    
for param in model_xray.projector_xray.parameters():
    param.requires_grad=True

    
for name, param in model_xray.fc.named_parameters():
    param.requires_grad=True
    
    
epochs = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.02)
scaler = GradScaler()
mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, eta_min = 3e-6)

model_xray = model_xray.to(device)

early_stopping = EarlyStopping(patience = 3)
best_val_score = -999

for epoch in range(epochs):
    tr_loss = 0
    model_xray.train()
    # fc_head.train()
    # Wrap your data loader with tqdm for a progress bar
    train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}")
    all_labels_train = torch.empty((0, 4)).cuda()
    all_probs_train = torch.empty((0, 4)).cuda()
    for step, (xray, labels) in enumerate(train_bar):
        xray, labels = xray.to(device), labels.to(device)
        
        optimizer.zero_grad()
        with autocast():
            out = model_xray(xray)
            loss = criterion(out, labels)
            
        # Backward and optimize with scaled gradients
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()
        prob_train = torch.sigmoid(out)
        all_labels_train = torch.cat((all_labels_train, labels), 0)
        all_probs_train = torch.cat((all_probs_train, prob_train), 0)
        
        tr_loss += loss.item()
        lr = optimizer.param_groups[0]['lr']
        train_bar.set_postfix({"Loss": loss.item(), "Lr": lr})
        
        # if step % 1000 == 0:
        #     wandb.log({'train loss': loss.item(), 'lr': lr})
        

    # Calculate average training loss for the epoch
    avg_train_loss = tr_loss / len(train_loader)
    # temp = criterion.temperature.item()
    
    all_labels_train = all_labels_train.detach().cpu().numpy()
    all_probs_train = all_probs_train.detach().cpu().numpy()  
    
    roc_auc_macro_train, roc_auc_micro_train = calc_roc_auc(all_labels_train, all_probs_train)
    map_train = calculate_multilabel_map(all_labels_train, all_probs_train) 
    auprc_train = average_precision_score(all_labels_train, all_probs_train, average="weighted")
    
    train_bar.set_description(f"Epoch {epoch + 1}/{epochs} - Avg Train Loss: {avg_train_loss}, LR: {lr}")
    
    print("\nMetrics after Train:")
    print("-" * 30)
    print(f"Roc-Prc: {auprc_train:.4f} | Map: {map_train:.4f} | Roc-Auc: {roc_auc_macro_train:.4f}")
    print("-" * 40)
    
    mainscheduler.step()
    # Validation loop with tqdm
    val_bar = tqdm(val_loader, desc=f"Validating Epoch {epoch + 1}/{epochs}")
    with torch.no_grad():
        val_loss = 0
        model_xray.eval()
        
        all_labels_val = torch.empty((0, 4)).cuda()
        all_probs_val = torch.empty((0, 4)).cuda()
        for xray, labels in val_bar:
            xray, labels = xray.to(device), labels.to(device)
 
            with autocast():
                out = model_xray(xray)
                loss = criterion(out, labels)
            
            prob_val = torch.sigmoid(out)
            all_labels_val = torch.cat((all_labels_val, labels), 0)
            all_probs_val = torch.cat((all_probs_val, prob_val), 0)
            
            val_loss += loss.item()
    val_loss_avg = val_loss / len(val_loader)
    
    all_labels_val = all_labels_val.detach().cpu().numpy()
    all_probs_val = all_probs_val.detach().cpu().numpy()  
    
    roc_auc_macro, roc_auc_micro = calc_roc_auc(all_labels_val, all_probs_val)
    map_val = calculate_multilabel_map(all_labels_val, all_probs_val) 
    auprc = average_precision_score(all_labels_val, all_probs_val, average="weighted")
    
    val_bar.set_description(f"Epoch {epoch + 1}/{epochs} - Avg Val Loss: {val_loss_avg}")
    
    print("\nMetrics after Validation:")
    print("-" * 30)
    print(f"Roc-Prc: {auprc:.4f} | Map: {map_val:.4f} | Roc-Auc: {roc_auc_macro:.4f}")
    print("-" * 40)
         
    # wandb.log({'Epoch': epoch, 'Train Loss Epoch': avg_train_loss, 'Val Loss Epoch': val_loss_avg, 'Learning Rate': lr})
    
    early_stopping(val_loss_avg)
    if early_stopping.early_stop:
        print(f'Early stopping at epoch {epoch} with val loss: {val_loss_avg}')
        break  # Break out of the loop
        
    if roc_auc_macro > best_val_score:
        best_val_score = roc_auc_macro
        torch.save(model_xray.state_dict(), './saved_models/best_multimodel.pth')

        plot_roc_curve(all_labels_val, all_probs_val, all_labels_val.shape[1], save_path=f'../plots/multimodal_infer/roc_curve_val.png')
        plot_precision_recall_curve(all_labels_val, all_probs_val, save_path=f'../plots/multimodal_infer/precision_recall_curve_val.png')

model_xray.load_state_dict(torch.load('./saved_models/best_multimodel.pth'))
with torch.no_grad():
    model_xray.eval()

    all_labels_test = torch.empty((0, 4)).cuda()
    all_probs_test = torch.empty((0, 4)).cuda()
    test_bar = tqdm(test_loader, desc="Test Bar")
    for xray, labels in test_bar:
        xray, labels = (
            xray.to(device), labels.to(device)
        )
        with autocast():
            out = model_xray(xray)
 

        prob_test = torch.sigmoid(out)
        all_labels_test = torch.cat((all_labels_test, labels), 0)
        all_probs_test = torch.cat((all_probs_test, prob_test), 0)


all_labels_test = all_labels_test.detach().cpu().numpy()
all_probs_test = all_probs_test.detach().cpu().numpy()  

roc_auc_macro_test, roc_auc_micro_test = calc_roc_auc(all_labels_test, all_probs_test)
map_test = calculate_multilabel_map(all_labels_test, all_probs_test) 
auprc_test = average_precision_score(all_labels_test, all_probs_test, average="weighted")


print("\nMetrics after Testing:")
print("-" * 30)
print(f"Roc-Prc: {auprc_test:.4f} | Map: {map_test:.4f} | Roc-Auc: {roc_auc_macro_test:.4f}")
print("-" * 40)

plot_roc_curve(all_labels_test, all_probs_test, all_labels_test.shape[1], save_path=f'../plots/multimodal_infer/roc_curve_test.png')
plot_precision_recall_curve(all_labels_test, all_probs_test, save_path=f'../plots/multimodal_infer/precision_recall_curve_test.png')