# <div align="center">DORAEMON: Deep Object Recognition And Embedding Model Of Networks</div>

<p align="center">
<img src="./misc/doraemon.jpg">
</p>

<p align="center">
<img src="https://img.shields.io/badge/python-3.10-blue.svg">
<img src="https://img.shields.io/badge/pytorch-2.0+-orange.svg">
<img src="https://img.shields.io/badge/torchmetrics-0.11.4-green.svg">
<img src="https://img.shields.io/badge/timm-0.9.16-red.svg">
<img src="https://img.shields.io/badge/opencv-4.7.0-lightgrey.svg">
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>

## 🚀 Quick Start

<summary><b>Installation Guide</b></summary>

```bash
# Create and activate environment
python -m venv doraemon
source doraemon/bin/activate

# Install Doraemon
pip install doraemon-torch

# If you need to install in editable mode (for development)
pip install -e .
```

## 📢 What's New

- 🎁 2025.03.16: Doraemon v0.0.3 released
- 🎁 2024.10.01: Content-Based Image Retrieval (CBIR): Training on a real Amazon product dataset with a complete pipeline for training, end-to-end validation, and visualization. Please check [README.md](models/faceX/README_CBIR.md)
- 🎁 2024.04.01: Face Recognition: Based on a cleaned face dataset with over 70,000 IDs and 3.6 million images, validated with LFW. Includes loss functions like ArcFace, CircleLoss, and MagFace.
- 🎁 2023.06.01: Image Classification (IC): Given the Oxford-IIIT Pet dataset. Supports different learning rates for different layers, hard example mining, multi-label and single-label training, bad case analysis, GradCAM visualization, automatic labeling to aid semi-supervised training, and category-specific data augmentation. Refer to [README.md](models/classifier/README.md)

## ✨ Highlights
- [Optimization Algorithms](doraemon/engine/optimizer.py): Various optimization techniques to enhance model training efficiency, including SGD, Adam, and SAM (Sharpness-Aware Minimization).

- [Data Augmentation](doraemon/dataset/transforms.py): A variety of data augmentation techniques to improve model robustness, such as CutOut, Color-Jitter, and Copy-Paste etc.

- [Regularization](doraemon/engine/optimizer.py): Techniques to prevent overfitting and improve model generalization, including Label Smoothing, OHEM, Focal Loss, and Mixup.

- [Visualization](doraemon/utils/cam.py): Integrated visualization tool to understand model decision-making, featuring GradCAM.

- [Personalized Data Augmentation](doraemon/built/class_augmenter.py): Apply exclusive data augmentation to specific classes with Class-Specific Augmentation.

- [Personalized Hyperparameter Tuning](doraemon/built/layer_optimizer.py): Apply different learning rates to specific layers using Layer-Specific Learning Rates.


## 📚 Tutorials

For detailed guidance on specific tasks, please refer to the following resources:

- **Image Classification**: If you are working on image classification tasks, please refer to [Doc: Image Classification](doraemon/models/classifier/README.md).

- **Image Retrieval**: For image retrieval tasks, please refer to [Doc: Image Retrieval](doraemon/models/representation/README_CBIR.md).

- **Face Recognition**: Stay tuned.

## 📊 Datasets

Doraemon integrates the following datasets, allowing users to quickly start training:

- **Image Retrieval**: Available at [Hugging Face Image Retrieval Dataset](https://huggingface.co/datasets/wuji3/image-retrieval)
- **Face Recognition**: Available at [Hugging Face Face Recognition Dataset](https://huggingface.co/datasets/wuji3/face-recognition)
- **Image Classification**: Available at [Oxford-IIIT Pet Dataset](https://huggingface.co/datasets/wuji3/oxford-iiit-pet)

## 🧩 Supported Models
 
**Doraemon** now supports 1000+ models through integration with Timm:
 
- All models from `timm.list_models(pretrained=True)`
- Including CLIP, SigLIP, DeiT, BEiT, MAE, EVA, DINO and more

[Model Performance Benchmarks](https://github.com/huggingface/pytorch-image-models/tree/main/results) can help you select the most suitable model by comparing:
- Inference speed
- Training efficiency 
- Accuracy across different datasets
- Parameter count vs performance trade-offs

> For detailed benchmark results, see [@huggingface/pytorch-image-models#1933](https://github.com/huggingface/pytorch-image-models/issues/1933)
