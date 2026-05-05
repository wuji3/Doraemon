# <div align="center">DORAEMON: Deep Object Recognition And Embedding Model Of Networks</div>

<p align="center">
<img src="./misc/doraemon.jpg">
</p>

<p align="center">
<img src="https://img.shields.io/badge/doraemon-0.0.10a-brightgreen.svg">
<img src="https://img.shields.io/badge/python-3.10-blue.svg">
<img src="https://img.shields.io/badge/pytorch-2.5.1%2B-orange.svg">
<img src="https://img.shields.io/badge/torchmetrics-0.11.4-green.svg">
<img src="https://img.shields.io/badge/timm-0.9.16-red.svg">
<img src="https://img.shields.io/badge/opencv-4.7.0-lightgrey.svg">
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>

**Doraemon** is a PyTorch-based visual modeling library for image classification, content-based image retrieval, face recognition, and model deployment. It integrates `timm` backbones, common metric-learning heads, training utilities, evaluation tools, visualization, and Hugging Face-compatible deployment APIs.

## Highlights

- [Training engine](doraemon/engine/vision_engine.py): Unified training flow for classification and embedding-based recognition tasks.
- [Optimization algorithms](doraemon/engine/optimizer.py): SGD, Adam, SAM, layer-specific learning rates, and related training strategies.
- [Data augmentation](doraemon/dataset/transforms.py): CutOut, ColorJitter, Copy-Paste, Mixup, and class-specific augmentation.
- [Regularization and losses](doraemon/engine/optimizer.py): Label smoothing, OHEM, Focal Loss, ArcFace, CircleLoss, MagFace, and more.
- [Visualization](doraemon/utils/cam.py): GradCAM-based model interpretation and bad-case analysis.
- [Deployment](deploy/README.md): Local inference and Hugging Face `AutoModel` / `AutoProcessor` integration.

## Installation

```bash
# Create and activate environment
python -m venv doraemon
source doraemon/bin/activate

# Install from PyPI
pip install doraemon-torch

# Or install in editable mode for development
pip install -e .
```

## Task Status

| Task | Status | Docs |
| --- | --- | --- |
| Image Classification | Supported | [Image Classification](doraemon/models/classifier/README.md) |
| Content-Based Image Retrieval | Supported | [Image Retrieval](doraemon/models/representation/README_CBIR.md) |
| Face Recognition | In progress | Training on MS-Celeb-1M-v1c and evaluation on LFW are supported internally; the end-to-end public pipeline is still being organized. |

## Tutorials

For task-specific data preparation, model configuration, training, evaluation, and visualization examples:

- **Image Classification**: [Doc: Image Classification](doraemon/models/classifier/README.md)
- **Image Retrieval**: [Doc: Image Retrieval](doraemon/models/representation/README_CBIR.md)
- **Face Recognition**: Stay tuned.

## Datasets

Doraemon provides dataset entry points for quickly starting experiments:

- **Image Classification**: [Oxford-IIIT Pet](https://huggingface.co/datasets/wuji3/oxford-iiit-pet)
- **Image Retrieval**: [Ecommerce Product](https://huggingface.co/datasets/wuji3/image-retrieval)
- **Face Recognition**: [MS-Celeb-1M-v1c](https://huggingface.co/datasets/wuji3/face-recognition)

## Supported Models

Doraemon supports 1000+ visual backbones through `timm`:

- All models from `timm.list_models(pretrained=True)`
- CLIP, SigLIP, DeiT, BEiT, MAE, EVA, DINO, ResNet, Swin Transformer, ViT, and more

[Model Performance Benchmarks](https://github.com/huggingface/pytorch-image-models/tree/main/results) can help select backbones by inference speed, training efficiency, accuracy, and parameter count.

> For detailed benchmark results, see [@huggingface/pytorch-image-models#1933](https://github.com/huggingface/pytorch-image-models/issues/1933).

## Deployment API

Doraemon supports deployment with a trained weight file plus model configuration and inference code:

- **Local inference**: Run a trained `*.pt` model with the deployment config.
- **Hugging Face integration**: Publish models that can be loaded with:
  - `AutoModel.from_pretrained()`
  - `AutoProcessor.from_pretrained()`

For detailed deployment instructions and examples, see the [Deployment Guide](deploy/README.md).

## What's New

- 2025.11.07: [Doraemon paper](https://arxiv.org/abs/2511.04394) released; welcome to <a href="#citation">cite our paper</a> if you find the project useful for your research or development.
- 2025.03.16: Doraemon v0.1.0 released.
- 2024.10.01: Content-Based Image Retrieval (CBIR) pipeline released with product data collected from Kaggle and TianChi. See [Image Retrieval](doraemon/models/representation/README_CBIR.md).
- 2024.04.01: Face Recognition support added with MS-Celeb-1M-v1c training data and LFW validation. Public end-to-end documentation is still in progress.
- 2023.06.01: Image Classification support released with Oxford-IIIT Pet examples, hard example mining, GradCAM visualization, auto-labeling, and class-specific augmentation. See [Image Classification](doraemon/models/classifier/README.md).

## Citation

<span id="citation"></span>

If you find **Doraemon** useful for your research or development, please cite the following <a href="https://arxiv.org/abs/2511.04394" target="_blank">paper</a>:

```bibtex
@misc{du2025visual,
      title={DORAEMON: A Unified Library for Visual Object Modeling and Representation Learning at Scale},
      author={Ke Du and Yimin Peng and Chao Gao and Fan Zhou and Siqiao Xue},
      year={2025},
      journal={arXiv preprint arXiv:2511.04394},
      url={https://arxiv.org/abs/2511.04394},
}
```
