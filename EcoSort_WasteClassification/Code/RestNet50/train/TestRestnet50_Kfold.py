import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ===== CONFIG =====
DATA_DIR = "D:/cpe_kps/Ai/dataset/dataset_subclass_Label/dataset_split_strict/test"  
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== TRANSFORM =====
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ===== LOAD TEST DATASET =====
test_dataset = datasets.ImageFolder(DATA_DIR, transform=val_tf)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
class_names = test_dataset.classes
num_classes = len(class_names)

# ===== LOAD 5 MODELS =====
def load_model(path):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

model_paths = [f"best_model_fold{i}.pth" for i in range(1, 6)]
models_ensemble = [load_model(p) for p in model_paths]

# ===== INFERENCE + ENSEMBLE =====
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        all_logits = []

        for model in models_ensemble:
            logits = model(imgs)
            all_logits.append(logits)

        avg_logits = torch.stack(all_logits).mean(dim=0)
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
plt.title("Confusion Matrix (Ensemble of 5 folds)")
plt.tight_layout()
plt.savefig("confusion_matrix_ensemble_Test.png")
plt.show()

# ===== CLASSIFICATION REPORT =====
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))
