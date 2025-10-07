# EcoSort Waste Classification — Credits

This document acknowledges the people, resources, and open-source projects that
make EcoSort Waste Classification possible.

## Project & Operations
- **Project ownership & primary development**: Alongkot (EcoSort Waste Classification team)
- **Documentation & deployment support**: Maintainers managing the README, tutorial notes, and Flask/ngrok setup guides

## Research & Inspiration
- *SOLID WASTE CLASSIFICATION USING MODIFIED RESNET-50 MODEL WITH TRANSFER LEARNING APPROACH*
- *EfficientNet-Based Deep Learning Model for Advanced Waste Classification*
- Additional architecture notes stored in `Resources/architecture_overview.txt` and `Resources/model_overview.md`

## Machine Learning Frameworks
- [PyTorch](https://pytorch.org/) and [torchvision](https://pytorch.org/vision/stable/) for model development
- [efficientnet-pytorch](https://github.com/lukemelas/EfficientNet-PyTorch) for EfficientNet-B1 implementation
- [scikit-learn](https://scikit-learn.org/) for stratified k-fold utilities and evaluation metrics
- [NumPy](https://numpy.org/) and [pandas](https://pandas.pydata.org/) for data processing

## Data Visualisation & Utilities
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for confusion matrices and training plots
- [tqdm](https://github.com/tqdm/tqdm) for progress monitoring during training

## Web Experience
- [Flask](https://flask.palletsprojects.com/) and Jinja2 templating for the inference UI
- [Pillow](https://python-pillow.org/) for image preprocessing
- [Werkzeug](https://werkzeug.palletsprojects.com/) utilities for secure file handling
- [ngrok](https://ngrok.com/) for optional remote access to the local web server

## Dataset Preparation
- Internal waste classification dataset (`dataset_subclass_Label/dataset_split_strict`) curated and labelled by the EcoSort team
- Stratified cross-validation pipeline ensuring balanced coverage across the 21 subclasses

## Acknowledgement
EcoSort expresses gratitude to the open-source community and researchers whose
work underpins this project. If you contribute improvements or new assets,
please add your details here when submitting pull requests.

