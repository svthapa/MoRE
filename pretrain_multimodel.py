import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import argparse
import time 

import sys
sys.path.append('./utils')
from ecg_augmentations import ECGAugmentor
from create_dataset import MultiModalData
from build_model import ViTModelEcg, ViTModelXray, ProjectionHead, MultiModal
from metrics import EarlyStopping

from info_nce import InfoNCE, info_nce

def load_data(filepath, test_size=0.02, random_state=42):
    data_new = np.load(filepath, allow_pickle=True)
    train_data = [item for item in data_new if item[5] == 'train']
    train_split, val_split = train_test_split(train_data, test_size=test_size, random_state=random_state)
    val_split = [item for item in data_new if item[5] == 'validate'] + val_split
    return train_split, val_split

def get_data_loaders(train_data, val_data, batch_size=100, num_workers=14):
    ecg_augmentor = ECGAugmentor()
    train_dataset = MultiModalData(train_data, ecg_augmentor)
    val_dataset = MultiModalData(val_data, ecg_augmentor, phase='val')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
    return train_loader, val_loader


def train_one_epoch(
    model, train_loader, optimizer, scaler, device, criterion, accumulation_steps, \
        epochs, epoch
):
    model.train()
    total_loss = 0
    train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}")
    for step, (xray, ecg, note_id, mask) in enumerate(train_bar):
        xray, ecg, note_id, mask = xray.to(device), ecg.to(device), \
                                    note_id.to(device), mask.to(device)
        with autocast():
            xray_out, xray_text, ecg_out, ecg_text, note_out = model(xray, ecg, note_id, mask)
            loss2 = (criterion(note_out, xray_text) + criterion(xray_text, note_out)) / 2
            loss3 = (criterion(note_out, ecg_text) + criterion(ecg_text, note_out)) / 2
            loss = (loss2 + loss3)
            
        scaler.scale(loss).backward()
        
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        total_loss += loss.item()
        lr = optimizer.param_groups[0]['lr']
        train_bar.set_postfix({"Loss": f"{loss.item():.4f}", 'LR': lr})
        
    return total_loss / len(train_loader)

def validate(
    model, val_loader, device, criterion, epochs, epoch
):
    model.eval()
    total_loss = 0
    val_bar = tqdm(val_loader, desc=f"Validating Epoch {epoch + 1}/{epochs}")
    with torch.no_grad():
        for xray, ecg, note_id, mask in val_bar:
            xray, ecg, note_id, mask = xray.to(device), ecg.to(device), note_id.to(device), mask.to(device)
            with autocast():
                xray_out, xray_text, ecg_out, ecg_text, note_out = model(xray, ecg, note_id, mask)
                loss2 = (criterion(note_out, xray_text) + criterion(xray_text, note_out)) / 2
                loss3 = (criterion(note_out, ecg_text) + criterion(ecg_text, note_out)) / 2
                loss = (loss2 + loss3)
            total_loss += loss.item()
            
            val_bar.set_postfix({"Val Loss": f"{loss.item():.4f}"})
    return total_loss / len(val_loader)

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain MoRE")
    parser.add_argument('--data_path', type=str, default='../data/path_to_data.npy', help='Path to the dataset file')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=14, help='Number of worker threads for data loading')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay for AdamW optimizer')
    parser.add_argument('--t_0', type=int, default=10, help='Number of iterations for the first restart in CosineAnnealingWarmRestarts')
    parser.add_argument('--eta_min', type=float, default=3e-6, help='Minimum learning rate in CosineAnnealingWarmRestarts')
    parser.add_argument('--temperature_initial', type=float, default=0.1, help='Initial temperature for InfoNCE loss')
    parser.add_argument('--learnable_temp', type=float, default=0.02, help='Learnable parameter temperature for InfoNCE')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--patience', type=int, default=10, help='Early Stopping patience')
    
    return parser.parse_args()

def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load data and initialize dataloaders
    train_data, val_data = load_data(args.data_path)
    train_loader, val_loader = get_data_loaders(train_data, val_data, \
                                    batch_size=args.batch_size, num_workers=args.num_workers)
    # Initialize model and move it to the appropriate device
    model = MultiModal().to(device)

    params = model.parameters()
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.t_0, eta_min=args.eta_min)
    # Setup loss function
    criterion = InfoNCE(args.temperature_initial)
    criterion.temperature = nn.Parameter(torch.tensor(args.learnable_temp))  # Make the temperature learnable
    scaler = GradScaler()
    accumulation_steps = args.accumulation_steps
    
    best_val_loss = 999
    early_stopping = EarlyStopping(patience = args.patience)
    # Main training and validation loop
    for epoch in range(args.epochs):
        start = time.time()
        print(f"Epoch {epoch+1}/{args.epochs}")

        # Training
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, \
                                     device, criterion, args.accumulation_steps, epochs, epoch)


        # Validation
        val_loss = validate(model, val_loader, device, criterion, epochs, epoch)

        # Learning rate scheduler step
        lr_scheduler.step()
        
        early_stopping(val_loss)
        
        if early_stopping.early_stop:
            print(f'Early stopping at epoch {epoch} with best val loss: {best_val_loss}')
            break  # Break out of the loop
            
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"best_multimodel_epoch_{epoch}_batch{args.batch_size}_loss_{val_loss:.4f}.pth")
            
        # Optionally save model
        if epoch % 5 == 0:  # save every 5 epochs
            torch.save(model.state_dict(), f"multimodel_epoch_{epoch}_batch{args.batch_size}.pth")
        
        lr = optimizer.param_groups[0]['lr']
        temp = criterion_simclr.temperature.item()
        
        print("\n" + "=" * 40)
        print(f"Epoch {epoch + 1}/{epochs} - Training Summary")
        print("-" * 40)
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Learning Rate: {lr:.7f}")
        print(f"Temperature: {temp:.4f}")
        print(f"Time taken: {(time.time() - start) / 60}")
        print("=" * 40 + "\n")

if __name__ == '__main__':
    main()