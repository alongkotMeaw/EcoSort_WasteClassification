# EcoSort Waste Classification

EcoSort Waste Classification is an ensemble-based computer vision project that
identifies 21 fine-grained waste subclasses and maps them to five household bin
categories (Recyclable, General, Biodegradable, Hazardous, and E-waste). The
system combines ResNet50 and EfficientNet-B1 training pipelines with a Flask web
application that exposes real-time image upload and camera capture support.

## Highlights
- 21-class dataset covering common recyclable, general, biodegradable, hazardous, and electronic waste.
- ResNet50 5-fold ensemble deployed for inference with optional class-specific confidence thresholds.
- EfficientNet-B1 5-fold pipeline with inverse-frequency class weighting for complementary evaluation.
- Flask web UI with drag-and-drop uploads, camera capture, and contextual disposal guidance.
- Documentation and resources that summarise architecture decisions, tuned thresholds, and reference studies.

## Repository Layout
```
Code/
  RestNet50/
    train/            # 5-fold ResNet50 training utilities
    output/           # Logs, checkpoints, confusion matrices
  EfficientB0/
    train/            # EfficientNet-B1 training and ensemble evaluation scripts
    output/           # Training logs and visualisations
  webpage/
    web/              # Flask application, static assets, templates, thresholds
Resources/            # Architecture notes and supporting research papers
requirements.txt      # Python dependencies with pinned versions
README.md             # Project documentation
```

## Prerequisites
- Python 3.10 (recommended for Torch 2.2.x compatibility)
- `pip` and a virtual environment tool such as `venv` or `conda`
- CUDA-capable GPU (optional but recommended for training)
- [ngrok](https://ngrok.com/) (optional) for sharing the web demo outside the local network

## Installation
```bash
git clone https://github.com/your-account/EcoSort_WasteClassification.git
cd EcoSort_WasteClassification
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Flask Web App
1. Collect the five ResNet50 checkpoints (`best_model_fold1.pth` ... `best_model_fold5.pth`)
   and place them in `Code/webpage/web/model/`. Update the `MODEL_PATHS` constant in
   `Code/webpage/web/main/main.py` if you store them elsewhere.
2. (Optional) review the threshold configuration in `Code/webpage/web/main/class_thresholds.json`.
3. Start the app from the project root:
   ```bash
   python Code/webpage/web/main/main.py
   ```
4. Open `http://127.0.0.1:5000/` in a browser. The console also prints a LAN URL.
5. To expose the interface externally, launch ngrok in a second terminal:
   ```bash
   ngrok http 5000
   ```

## Training Pipelines

### ResNet50 Ensemble
1. Point `DATA_DIR` in `Code/RestNet50/train/train_kfold_resnet50.py` to the
   directory that holds one folder per class (no need to pre-split the data).
2. Activate your environment and run:
   ```bash
   python Code/RestNet50/train/train_kfold_resnet50.py
   ```
3. Outputs include:
   - `best_model_fold{n}.pth` checkpoints
   - `log_folds/fold{n}_log.csv` training history
   - `conf_matrix_fold{n}.png` per-fold confusion matrices

### EfficientNet-B1 Weighted Ensemble
1. Update the `DATA_DIR` constant in
   `Code/EfficientB0/train/train_kfold_log_b3_mixup_weighted.py` to the same
   single dataset directory.
2. Run the training script:
   ```bash
   python Code/EfficientB0/train/train_kfold_log_b3_mixup_weighted.py
   ```
3. Per-fold checkpoints, logs, and confusion matrices are written to
   `Code/EfficientB0/output/`.

### Ensemble Evaluation
Use `Code/EfficientB0/train/test_ensemble_efficientnet.py` to evaluate the
EfficientNet ensemble on the hold-out `test/` split and generate reports:
```bash
python Code/EfficientB0/train/test_ensemble_efficientnet.py
```

## Threshold Tuning & UI Behaviour
- `Code/webpage/web/main/class_thresholds.json` holds class-level probability
  thresholds. When a prediction exceeds its threshold it is considered confident
  enough to display; otherwise the system falls back to the highest probability.
- `Code/webpage/web/main/tuned_thresholds.py` is an alternative Flask entry point
  with Thai localisation for waste group labels and detailed probability tables.
- Waste subclasses are mapped to bin categories through the `GROUPS` structure
  inside `Code/webpage/web/main/main.py`, which in turn surfaces tips drawn from
  `Code/webpage/web/main/info_txt/`.

## Dataset Expectations
```
dataset_root/
  AluminumCan/
    img001.jpg
    ...
  Battery/
    ...
  ...
```
All folders should contain RGB images. ResNet50 scripts resize inputs to
224x224, while EfficientNet scripts resize to 260x260. Augmentations include
random horizontal flip, colour jitter, and ImageNet normalisation. Optionally,
you can keep aside a held-out test split to run `test_ensemble_efficientnet.py`
or other evaluations after cross-validation.

## Project Credits
- Development & Maintenance: EcoSort Waste Classification project team (repository owner: Alongkot)
- Research References (stored in `Resources/`):
  - *SOLID WASTE CLASSIFICATION USING MODIFIED RESNET-50 MODEL WITH TRANSFER LEARNING APPROACH*
  - *EfficientNet-Based Deep Learning Model for Advanced Waste Classification*
- Machine Learning Frameworks: PyTorch, torchvision, EfficientNet-PyTorch
- Web Interface: Flask, Jinja2, and Bootstrap-inspired styling
- Dataset Preparation: Stratified fold splits of the internal `dataset_subclass_Label/dataset_split_strict` waste dataset

## License
Specify the project license before publishing the repository publicly (e.g. MIT, Apache-2.0, GPL-3.0).

## 📜 Credits

This document acknowledges the people, resources, and open-source projects that
make EcoSort Waste Classification possible.

### Project & Operations
- **Project ownership & primary development**: Alongkot (EcoSort Waste Classification team)
- **Documentation & deployment support**: Maintainers managing the README, tutorial notes, and Flask/ngrok setup guides

### Research & Inspiration
- *SOLID WASTE CLASSIFICATION USING MODIFIED RESNET-50 MODEL WITH TRANSFER LEARNING APPROACH*
- *EfficientNet-Based Deep Learning Model for Advanced Waste Classification*
- Additional architecture notes stored in `Resources/architecture_overview.txt` and `Resources/model_overview.md`

### Machine Learning Frameworks
- [PyTorch](https://pytorch.org/) and [torchvision](https://pytorch.org/vision/stable/) for model development
- [efficientnet-pytorch](https://github.com/lukemelas/EfficientNet-PyTorch) for EfficientNet-B1 implementation
- [scikit-learn](https://scikit-learn.org/) for stratified k-fold utilities and evaluation metrics
- [NumPy](https://numpy.org/) and [pandas](https://pandas.pydata.org/) for data processing

### Data Visualisation & Utilities
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for confusion matrices and training plots
- [tqdm](https://github.com/tqdm/tqdm) for progress monitoring during training

### Web Experience
- [Flask](https://flask.palletsprojects.com/) and Jinja2 templating for the inference UI
- [Pillow](https://python-pillow.org/) for image preprocessing
- [Werkzeug](https://werkzeug.palletsprojects.com/) utilities for secure file handling
- [ngrok](https://ngrok.com/) for optional remote access to the local web server

### Dataset Preparation
- Internal waste classification dataset (`dataset_subclass_Label/dataset_split_strict`) curated and labelled by the EcoSort team
- Stratified cross-validation pipeline ensuring balanced coverage across the 21 subclasses

### Acknowledgement
EcoSort expresses gratitude to the open-source community and researchers whose
work underpins this project. If you contribute improvements or new assets,
please add your details here when submitting pull requests.
