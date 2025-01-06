# <div align="center">DORAEMON (Deep Object Recognition And Embedding Model Of Networks)</div>

<p align="center">
<img src="misc/doraemon.jpg">
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

<details>
<summary><b>Installation Guide</b></summary>

```bash
# Create and activate environment
conda create -n Doraemon python=3.10 && conda activate Doraemon

# Install PyTorch (CUDA or CPU version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
# or
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install dependencies
pip install -r requirements.txt

# For CBIR functionality
conda install faiss-gpu=1.8.0 -c pytorch -y

# Optional: Install Arial font for faster inference
mkdir -p ~/.config/Doraemon && cp misc/Arial.ttf ~/.config/Doraemon
```
</details>

## 📢 What's New

- **[Oct. 2024]** [Content-Based Image Retrieval(CBIR)](models/faceX/README.md) support added with ConvNext backbone
- **[Apr. 2024]** [Face Recognition Task(FRT)](models/faceX/README.md) launched with various backbones and loss functions
- **[Jun. 2023]** [Image Classification Task(ICT)](models/classifier/README.md) released with advanced training strategies
- **[May. 2023]** Initial release of VisionDK

## 🧠 Implemented Methods

| Category | Methods |
|----------|---------|
| Optimization | SAM, Progressive Learning, OHEM, Focal Loss, Cosine Annealing |
| Regularization | Label Smoothing, Mixup, CutOut |
| Attention & Visualization | Attention Pool, GradCAM |
| Face Recognition | ArcFace, CircleLoss, MegFace, MV Softmax |

## 📚 Supported Models
 
Doraemon now supports 1000+ models through integration with TorchVision and Timm:
 
- **TorchVision Models** (100+)
  - All models from `torchvision.models.list_models()`
  - Including MobileNet, ShuffleNet, ResNet, ConvNext, EfficientNet, Swin, ViT and more
 
- **Timm Models** (1000+)
  - All models from `timm.list_models(pretrained=True)`
  - Including CLIP, SigLIP, DeiT, BEiT, MAE, EVA, DINO and more
 
For detailed model usage, please refer to [Image Classification Guide](models/classifier/README.md)

## 🛠️ Utility Tools

| Tool | Description | Usage |
|------|-------------|-------|
| Data Splitter | Split dataset into train/val sets | `python tools/data_prepare.py --postfix <jpg\|png> --root <path> --frac <ratio>` |
| Query-Gallery Prep | Prepare data for image retrieval | `python tools/build_querygallery.py --src <path> --frac <ratio>` |
| Augmentation Visualizer | Visualize data augmentations | `python -m tools.test_augment` |
| Data Deduplicator | Remove duplicate entries | `python tools/deduplicate.py` |

## 🤝 Contribute

- For contributions: Submit a pull request
- For questions or issues: Open an issue