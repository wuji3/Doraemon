# <div align="center">Image Classification</div>

## 🚀 Quick Start

### 1. Data Preparation

VisionDK supports both local datasets and HuggingFace datasets:

#### Local Datasets
We support two formats for local data:

**A. Single-label Classification (Folder Structure)**

Organize your data in the following structure:
```
data/pet/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── class2/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── val/
    ├── class1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── class2/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

We provide the Oxford-IIIT Pet Dataset as an example, which contains 37 pet breeds with ~200 images per category:
- Download: [Official URL](https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz) or [Baidu Cloud(Recommend)](https://pan.baidu.com/s/1PjM6kPoTyzNYPZkpmDoC6A) (Code: yjsl)
- Prepare the dataset:
  ```bash
  cd data
  tar -xf oxford-iiit-pet.tgz
  python split2dataset.py  # This will organize the data into train/val splits
  ```

**B. Multi-label Classification (CSV Format)**

We use CSV format for multi-label tasks. See our [Sample CSV File](../../data/toy-multi-cls.csv) for a complete example:
```csv
image_path,tag1,tag2,tag3,tag4,tag5,train
/path/to/image1.jpg,0,1,0,0,0,True
/path/to/image2.jpg,0,1,0,0,0,True
/path/to/image3.jpg,1,1,0,0,0,False
```
- `image_path`: Absolute path to image
- `tag1-tagN`: Binary labels (0 or 1)
- `train`: Boolean field (True for training set, False for validation set)

#### HuggingFace Datasets
We also support HuggingFace datasets (single-label only):
```yaml
data:
  root: dataset_name/config  # e.g., StarQuestLab/oxford-iiit-pet
```

#### Data Preparation Tool
For local single-label datasets, you can use our tool to split train/val sets:
```bash
python tools/data_prepare.py --postfix <jpg|png> --root <data_path> --frac <train_set_ratio>
```

### 2. Configuration

Modify the configuration file according to your task:
```yaml
# configs/classification/config.yaml
data:
  # Local datasets:
  root: data/pet                # For single-label (folder structure)
  root: data/multi_label.csv    # For multi-label (CSV format)
  
  # HuggingFace dataset (single-label only):
  root: dataset_name/config     # e.g., StarQuestLab/oxford-iiit-pet
```

### 3. Training

```bash
# Single GPU
python main.py --cfgs configs/classification/pet.yaml

# Multi-GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 main.py --cfgs configs/classification/pet.yaml

# Optional:
#   --sync_bn    # Sync BatchNorm layers
#   --resume     # Resume from checkpoint
#   --load_from  # Fine-tune from pretrained
```
Training logs will be saved to `run/exp/log{timestamp}.log`, containing:
- Dataset statistics (class distribution, image counts)
- Training progress (loss curves, learning rates)
- Validation metrics (accuracy, precision, recall)
- Commands for visualization and evaluation
```bash
# View training log
vi run/exp/log{timestamp}.log  # e.g., log20241113-155144.log
```

### 4. Evaluation & Visualization

After training, you can use the following commands for evaluation and visualization:

#### Visualization
```bash
# Basic usage
python visualize.py --cfgs <config.yaml> \
                   --weight <best.pt> \
                   --class_json <class_indices.json> \
                   --data <val_data_path> \
                   --target_class <class_name> \
                   --ema

# Optional arguments:
#   --cam           # Enable CAM visualization for model interpretability
#   --badcase       # Group misclassified images into a separate folder
#   --sampling N    # Visualize N random samples (e.g., --sampling 100)
#   --remove_label  # Hide prediction text on images
```

The visualization results will be saved to `visualization/exp/`:
- For single-label classification:
  - Normal mode: Shows top-5 predictions with confidence scores
  - CAM mode: Additional heatmap showing model's focus areas
  - Badcase mode: Automatically groups misclassified images
- For multi-label classification:
  - Shows predictions above threshold for each class
  - Supports per-class thresholds

#### Validation
```bash
# Evaluate model performance
python validate.py --cfgs <config.yaml> \
                  --weight <best.pt> \
                  --eval_topk 5 \
                  --ema

# The script will output:
# - Accuracy metrics (Top-1, Top-5 for single-label)
# - Per-class precision/recall (for multi-label)
# - Confusion matrix (optional, for single-label)
```

## 📊 Results

<p align="center">
  <img src="../../misc/visual&validation.jpg" width="40%">
  <br>
  <em>Left: CAM visualization showing model's attention. Right: Validation metrics.</em>
</p>