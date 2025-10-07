# Model Overview

## Ensemble Summary

- Primary backbone: ResNet50 trained from scratch with 5-fold stratified CV (21 classes).
- Secondary backbone: EfficientNet-B1 (random init) trained with weighted cross-entropy and mixup for 5 folds.
- Web inference uses the ResNet50 fold ensemble to drive UI predictions.

## Validation Highlights

- ResNet50 folds: 97.7%-98.3% validation accuracy, macro-F1 ~0.98.
- EfficientNet-B1 folds: 79%-82% validation accuracy, macro-F1 ~0.81.

## Training Notes

- Dataset root expected at `D:/cpe_kps/Ai/dataset/dataset_subclass_Label/dataset_split_strict` with `train/`, `val/`, and `test/` splits.
- Augmentations: resize to 224x224 (ResNet) or 260x260 (EfficientNet), random horizontal flip, color jitter, ImageNet normalization.
- Optimizer: Adam with learning rate 1e-4, batch size 32, early stopping patience 5.
- Class balancing: EfficientNet leverages inverse-frequency class weights.

## Inference Pipeline

1. Load five ResNet50 checkpoints and average softmax probabilities.
2. Aggregate probabilities to select the most plausible label.
3. Map the predicted class to a high-level waste group for UI display and guidance.
