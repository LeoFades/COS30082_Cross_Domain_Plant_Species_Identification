# -*- coding: utf-8 -*-
"""
ResNet50 PyTorch Training - Cross Domain Plant Species Identification
Converted from TensorFlow version for HuggingFace deployment
"""

import os
import time
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ==========================================
# Configuration
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Paths - UPDATE THESE
BASE_DIR = '/content/drive/MyDrive/AML_Group_Project'
ZIP_PATH = os.path.join(BASE_DIR, 'AML_project_herbarium_dataset.zip')
EXTRACT_DIR = '/content/AML_project_herbarium_dataset'
SAVE_DIR = os.path.join(BASE_DIR, 'Approach_1/Phyllis/models_pytorch')
os.makedirs(SAVE_DIR, exist_ok=True)

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS_FROZEN = 20
EPOCHS_UNFROZEN = 50
LR_FROZEN = 1e-3
LR_UNFROZEN = 1e-4

# ==========================================
# Extract Dataset
# ==========================================
if not os.path.exists(EXTRACT_DIR):
    print(f"Extracting dataset...")
    start = time.time()
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print(f"Extraction complete in {time.time()-start:.1f}s")
else:
    print("Dataset already extracted")

# Paths
train_txt = os.path.join(EXTRACT_DIR, 'list/train.txt')
test_txt = os.path.join(EXTRACT_DIR, 'list/test.txt')
groundtruth_txt = os.path.join(EXTRACT_DIR, 'list/groundtruth.txt')
paired_file = os.path.join(EXTRACT_DIR, 'list/class_with_pairs.txt')
unpaired_file = os.path.join(EXTRACT_DIR, 'list/class_without_pairs.txt')
species_list = os.path.join(EXTRACT_DIR, 'list/species_list.txt')

# ==========================================
# Load Dataset
# ==========================================
train_df = pd.read_csv(train_txt, sep=' ', names=['path', 'label'])
gt_df = pd.read_csv(groundtruth_txt, sep=' ', names=['path', 'label'])
test_df = pd.read_csv(test_txt, names=['path'])
test_df = pd.merge(test_df, gt_df, on='path', how='left')

# Remap labels to 0..N-1
unique_labels = sorted(train_df['label'].unique())
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
train_df['label'] = train_df['label'].map(label_to_idx)
test_df['label'] = test_df['label'].map(label_to_idx)

NUM_CLASSES = len(unique_labels)
print(f"Number of classes: {NUM_CLASSES}")
print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# Save label mapping for deployment
torch.save(label_to_idx, os.path.join(SAVE_DIR, 'label_to_idx.pt'))

# Load paired/unpaired classes
with open(paired_file, 'r') as f:
    paired_classes = [label_to_idx[int(line.strip())] for line in f if line.strip().isdigit() and int(line.strip()) in label_to_idx]
with open(unpaired_file, 'r') as f:
    unpaired_classes = [label_to_idx[int(line.strip())] for line in f if line.strip().isdigit() and int(line.strip()) in label_to_idx]

print(f"Paired classes: {len(paired_classes)}, Unpaired classes: {len(unpaired_classes)}")

# ==========================================
# Dataset Class
# ==========================================
class PlantDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['path'])
        image = Image.open(img_path).convert('RGB')
        label = row['label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ==========================================
# Transforms
# ==========================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),  # RandomZoom equivalent
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# DataLoaders
# ==========================================
train_dataset = PlantDataset(train_df, EXTRACT_DIR, train_transform)
test_dataset = PlantDataset(test_df, EXTRACT_DIR, test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

# ==========================================
# Model Definition (matches your TF architecture)
# ==========================================
class PlantResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load pretrained ResNet50
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Replace classifier head to match your TF architecture:
        # GlobalAveragePooling2D -> Dropout(0.3) -> Dense(256, relu, L2) -> BN -> Dropout(0.3) -> Dense(100)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze all backbone layers except the classifier head"""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    
    def unfreeze_last_n_layers(self, n=30):
        """Unfreeze last n layers of backbone"""
        # First freeze everything except fc
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        
        # Then unfreeze last n layers
        all_params = list(self.backbone.named_parameters())
        for name, param in all_params[-n:]:
            param.requires_grad = True
        
        # Always keep fc trainable
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable params: {trainable:,} / {total:,}")

# ==========================================
# Training Functions
# ==========================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
    
    return running_loss / total, 100. * correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_top5 = []
    
    for images, labels in tqdm(loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Top-5
        _, top5_pred = outputs.topk(5, dim=1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_top5.extend(top5_pred.cpu().numpy())
    
    # Top-5 accuracy
    top5_correct = sum(1 for i, label in enumerate(all_labels) if label in all_top5[i])
    top5_acc = 100. * top5_correct / total
    
    return running_loss / total, 100. * correct / total, top5_acc, np.array(all_preds), np.array(all_labels), np.array(all_top5)

# ==========================================
# Initialize Model
# ==========================================
model = PlantResNet50(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()

print(model)

# ==========================================
# Stage 1: Frozen Backbone Training
# ==========================================
print("\n" + "="*60)
print("Stage 1: Training with Frozen Backbone")
print("="*60)

model.freeze_backbone()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_FROZEN)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [], 'val_top5': []
}

best_val_acc = 0
patience_counter = 0
patience = 3

for epoch in range(1, EPOCHS_FROZEN + 1):
    print(f"\nEpoch {epoch}/{EPOCHS_FROZEN}")
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc, val_top5, _, _, _ = evaluate(model, test_loader, criterion, DEVICE)
    
    scheduler.step(val_loss)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_top5'].append(val_top5)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Top-5: {val_top5:.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'resnet50_frozen_best.pth'))
        print(f"Saved best model (acc: {best_val_acc:.2f}%)")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

frozen_epochs = len(history['train_loss'])

# ==========================================
# Stage 2: Fine-tuning with Unfrozen Layers
# ==========================================
print("\n" + "="*60)
print("Stage 2: Fine-tuning with Last 30 Layers Unfrozen")
print("="*60)

# Load best frozen model
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'resnet50_frozen_best.pth')))
model.unfreeze_last_n_layers(30)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_UNFROZEN)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-7)

best_val_acc = 0
patience_counter = 0
patience = 7

for epoch in range(1, EPOCHS_UNFROZEN + 1):
    print(f"\nEpoch {epoch}/{EPOCHS_UNFROZEN}")
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc, val_top5, _, _, _ = evaluate(model, test_loader, criterion, DEVICE)
    
    scheduler.step(val_loss)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_top5'].append(val_top5)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Top-5: {val_top5:.2f}%")
    print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'resnet50_plant.pth'))
        print(f"Saved best model (acc: {best_val_acc:.2f}%)")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# ==========================================
# Plot Training Curves
# ==========================================
plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Val Accuracy')
plt.axvline(x=frozen_epochs-1, color='red', linestyle='--', label='Unfreeze last 30 layers')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.axvline(x=frozen_epochs-1, color='red', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'resnet50_training_curves.png'), dpi=300)
plt.show()

# ==========================================
# Final Evaluation
# ==========================================
print("\n" + "="*60)
print("Final Evaluation")
print("="*60)

model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'resnet50_plant.pth')))
_, val_acc, val_top5, y_pred, y_true, y_top5 = evaluate(model, test_loader, criterion, DEVICE)

# Per-class accuracy
per_class_acc = {}
for c in range(NUM_CLASSES):
    mask = y_true == c
    if mask.sum() > 0:
        per_class_acc[c] = (y_pred[mask] == y_true[mask]).mean()

avg_per_class = np.mean(list(per_class_acc.values()))

print(f"\n=== Overall Test Set Accuracy ===")
print(f"Top-1 Accuracy: {val_acc:.2f}%")
print(f"Top-5 Accuracy: {val_top5:.2f}%")
print(f"Average Per-Class Accuracy: {avg_per_class*100:.2f}%")

# Paired vs Unpaired
def compute_subset_metrics(y_true, y_pred, y_top5, subset_classes):
    mask = np.isin(y_true, subset_classes)
    if mask.sum() == 0:
        return 0, 0, 0
    
    y_true_s = y_true[mask]
    y_pred_s = y_pred[mask]
    y_top5_s = y_top5[mask]
    
    top1 = (y_true_s == y_pred_s).mean()
    top5_hits = sum(1 for i, label in enumerate(y_true_s) if label in y_top5_s[i])
    top5 = top5_hits / len(y_true_s)
    
    subset_accs = []
    for c in subset_classes:
        mask_c = y_true_s == c
        if mask_c.sum() > 0:
            subset_accs.append((y_pred_s[mask_c] == y_true_s[mask_c]).mean())
    
    avg_class = np.mean(subset_accs) if subset_accs else 0
    return top1, top5, avg_class

paired_top1, paired_top5, paired_avg = compute_subset_metrics(y_true, y_pred, y_top5, paired_classes)
unpaired_top1, unpaired_top5, unpaired_avg = compute_subset_metrics(y_true, y_pred, y_top5, unpaired_classes)

print(f"\n=== Paired Classes Accuracy ===")
print(f"Top-1: {paired_top1*100:.2f}%, Top-5: {paired_top5*100:.2f}%, Avg Per-Class: {paired_avg*100:.2f}%")

print(f"\n=== Unpaired Classes Accuracy ===")
print(f"Top-1: {unpaired_top1*100:.2f}%, Top-5: {unpaired_top5*100:.2f}%, Avg Per-Class: {unpaired_avg*100:.2f}%")

# ==========================================
# Confusion Matrix
# ==========================================
cm = confusion_matrix(y_true, y_pred, normalize='true')
per_class_acc_arr = np.diag(cm)
sorted_indices = np.argsort(per_class_acc_arr)
worst20_idx = sorted_indices[:20]
best20_idx = sorted_indices[-20:]

real_labels = [idx_to_label[i] for i in range(NUM_CLASSES)]

# Best 20
plt.figure(figsize=(12, 10))
sns.heatmap(cm[np.ix_(best20_idx, best20_idx)],
            xticklabels=[real_labels[i] for i in best20_idx],
            yticklabels=[real_labels[i] for i in best20_idx],
            cmap='Blues', annot=False)
plt.title("Confusion Matrix - Top 20 Best Classes")
plt.xlabel("Predicted Class ID")
plt.ylabel("True Class ID")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'resnet50_cm_best20.png'), dpi=300)
plt.show()

# Worst 20
plt.figure(figsize=(12, 10))
sns.heatmap(cm[np.ix_(worst20_idx, worst20_idx)],
            xticklabels=[real_labels[i] for i in worst20_idx],
            yticklabels=[real_labels[i] for i in worst20_idx],
            cmap='Reds', annot=False)
plt.title("Confusion Matrix - Top 20 Worst Classes")
plt.xlabel("Predicted Class ID")
plt.ylabel("True Class ID")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'resnet50_cm_worst20.png'), dpi=300)
plt.show()

# Classification Report
report = classification_report(
    [idx_to_label[i] for i in y_true],
    [idx_to_label[i] for i in y_pred],
    digits=3, output_dict=True, zero_division=0
)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(SAVE_DIR, 'resnet50_classification_report.csv'))
print(f"\nSaved classification report to {SAVE_DIR}")

print("\n" + "="*60)
print("Training Complete!")
print(f"Best model saved to: {os.path.join(SAVE_DIR, 'resnet50_plant.pth')}")
print("="*60)
