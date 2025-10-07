# test_ensemble_efficientnet.py

import os
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ===== CONFIG =====
DATA_DIR = r"D:/cpe_kps/Ai/dataset/dataset_subclass_Label/dataset_split_strict/test"
MODEL_NAME = "efficientnet-b1"
MODEL_PATHS = [f"best_model_fold{i}.pth" for i in range(1, 5)]  # 4 โมเดล
BATCH_SIZE = 32
IMG_SIZE = 260
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== TRANSFORM (จาก train_kfold_log_b3_mixup_weighted.py) =====
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ===== LOAD TEST DATASET =====
test_dataset = datasets.ImageFolder(DATA_DIR, transform=val_tf)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
class_names = test_dataset.classes
num_classes = len(class_names)

# ===== LOAD MODELS =====
def load_model(path):
    model = EfficientNet.from_name(MODEL_NAME)
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

models_ensemble = [load_model(p) for p in MODEL_PATHS]

# ===== ENSEMBLE PREDICTION =====
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        logits_list = [model(imgs) for model in models_ensemble]
        avg_logits = torch.stack(logits_list).mean(dim=0)
        preds = avg_logits.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# ===== CONFUSION MATRIX =====
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (EfficientNet Ensemble Fold1–4)")
plt.tight_layout()
plt.savefig("conf_matrix_ensemble_b1_fold1to4_Test.png")
plt.show()

# ===== REPORT =====
print("Classification Report (Fold1–4 Ensemble):")
print(classification_report(all_labels, all_preds, target_names=class_names))
