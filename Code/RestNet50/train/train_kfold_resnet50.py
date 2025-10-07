# train_kfold_log.py
import os
import numpy as np
import pandas as pd
import torch
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ===== CONFIG =====
DATA_DIR = r"D:\cpe_kps\Ai\dataset\dataset_subclass_Label\dataset_split_strict"
NUM_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 32
PATIENCE = 5
LR = 1e-4
LOG_DIR = "log_folds"
os.makedirs(LOG_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== TRANSFORM =====
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# ===== DATASET =====
class SubsetWithTransform(Subset):
    """Subset that applies a transform on-the-fly without mutating the base dataset."""

    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        image, target = self.dataset[self.indices[idx]]
        if self.transform:
            image = self.transform(image)
        return image, target


full_dataset = ImageFolder(DATA_DIR, transform=None)

labels = full_dataset.targets
classes = full_dataset.classes
idx_to_label = {i: name for i, name in enumerate(classes)}
num_classes = len(classes)

# ===== K-FOLD =====
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
    print(f"\n Fold {fold}/{NUM_FOLDS}")
    train_set = SubsetWithTransform(full_dataset, train_idx, train_tf)
    val_set = SubsetWithTransform(full_dataset, val_idx, val_tf)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0
    patience = 0
    log_rows = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, targets in tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch}"):
            imgs, targets = imgs.to(device), targets.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, targets)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            pred = out.argmax(1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
        train_acc = correct / total
        train_loss = total_loss / len(train_loader)

        # ===== VAL =====
        model.eval()
        correct, total, val_loss = 0, 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                out = model(imgs)
                loss = loss_fn(out, targets)
                val_loss += loss.item()
                pred = out.argmax(1)
                correct += (pred == targets).sum().item()
                total += targets.size(0)
                y_true += targets.cpu().tolist()
                y_pred += pred.cpu().tolist()

        val_acc = correct / total
        val_loss /= len(val_loader)
        f1 = f1_score(y_true, y_pred, average='macro')

        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "f1_macro": f1
        })

        print(f" Epoch {epoch} | "
      f"Train Loss: {train_loss:.4f} | "
      f"Val Loss: {val_loss:.4f} | "
      f"Train Acc: {train_acc:.4f} | "
      f"Val Acc: {val_acc:.4f} | "
      f"F1: {f1:.4f}")


        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"best_model_fold{fold}.pth")
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(" Early stopping.")
                break

    # ===== LOG SAVE =====
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(os.path.join(LOG_DIR, f"fold{fold}_log.csv"), index=False)

    # ===== CONFUSION MATRIX =====
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix Fold {fold}")
    plt.tight_layout()
    plt.savefig(f"conf_matrix_fold{fold}.png")
    plt.close()
