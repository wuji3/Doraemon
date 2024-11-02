# <div align="center">VisionDK: Image Classification & Representation Learning ToolBox</div>

## 🚀 Quick Start

<details>
<summary><b>Installation Guide</b></summary>

```bash
# Create and activate environment
conda create -n vision python=3.10 && conda activate vision

# Install PyTorch (CUDA or CPU version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
# or
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install dependencies
pip install -r requirements.txt

# For CBIR functionality
conda install faiss-gpu=1.8.0 -c pytorch

# Optional: Install Arial font for faster inference
mkdir -p ~/.config/DuKe && cp misc/Arial.ttf ~/.config/DuKe
```
</details>

## 📢 What's New

- **[Oct. 2024]** Content-Based Image Retrieval (CBIR) support added with ConvNext backbone
- **[Apr. 2024]** Face Recognition Task (FRT) launched with various backbones and loss functions
- **[Jun. 2023]** Image Classification Task (ICT) released with advanced training strategies
- **[May. 2023]** Initial release of VisionDK

## 🧠 Implemented Methods

| Category | Methods |
|----------|---------|
| Optimization | SAM, Progressive Learning, OHEM, Focal Loss, Cosine Annealing |
| Regularization | Label Smoothing, Mixup, CutOut |
| Attention & Visualization | Attention Pool, GradCAM |
| Face Recognition | ArcFace, CircleLoss, MegFace, MV Softmax |

## 📚 Supported Models

| Model Family | Variants |
|--------------|----------|
| MobileNet | v2, v3_small, v3_large |
| ShuffleNet | v2_x0_5, v2_x1_0, v2_x1_5, v2_x2_0 |
| ResNet & ResNeXt | 18, 34, 50, 101, 152, 50_32x4d, 101_32x8d, 101_64x4d |
| ConvNext | tiny, small, base, large |
| EfficientNet | b0-b7, v2_s, v2_m, v2_l |
| Swin Transformer | tiny, small, base (v1 & v2) |

## 🛠️ Utility Tools

| Tool | Description | Usage |
|------|-------------|-------|
| Data Splitter | Split dataset into train/val sets | `python tools/data_prepare.py --postfix <jpg\|png> --root <path> --frac <ratio>` |
| Query-Gallery Prep | Prepare data for image retrieval | `python tools/build_querygallery.py --src <path> --frac <ratio>` |
| Augmentation Visualizer | Visualize data augmentations | `python -m tools.test_augment` |
| Data Deduplicator | Remove duplicate entries | `python tools/deduplicate.py` |

<p align="center">
  <img src="misc/augments.jpg" width="60%" alt="Data Augmentation Visualization">
</p>

## 🤝 Contribute

- For contributions: Submit a pull request
- For questions or issues: Open an issue