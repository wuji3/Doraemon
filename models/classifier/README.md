# <div align="center">Image Classification</div>

## 📦 Data Preparation

### Supported Data Formats

#### 1. Local Single-label Dataset
Organize your data in the following structure:
```
data/pet/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── ...
│   └── class2/
│       ├── image1.jpg
│       └── ...
└── val/
    ├── class1/
    │   ├── image1.jpg
    │   └── ...
    └── class2/
        ├── image1.jpg
        └── ...
```

**Example Dataset**: Oxford-IIIT Pet Dataset (37 pet breeds, ~200 images/class)
- Download: [Official URL](https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz) or [Baidu Cloud](https://pan.baidu.com/s/1PjM6kPoTyzNYPZkpmDoC6A) (Code: yjsl)
- Preparation:
  ```bash
  cd data
  tar -xf oxford-iiit-pet.tgz
  python split2dataset.py
  ```

#### 2. Local Multi-label Dataset (CSV)
See our [Sample CSV File](../../data/toy-multi-cls.csv):
```csv
image_path,tag1,tag2,tag3,tag4,tag5,train
/path/to/image1.jpg,0,1,0,0,0,True
/path/to/image2.jpg,0,1,0,0,0,True
```

#### 3. HuggingFace Dataset
```yaml
data:
  root: StarQuestLab/oxford-iiit-pet  # Format: {username}/{dataset_name}
```

### Data Preparation Tool
For local single-label datasets:
```bash
python tools/data_prepare.py --postfix <jpg|png> --root <data_path> --frac <train_set_ratio>
```

## 🚀 Training

### Configuration
```yaml
data:
  # Choose ONE of the following:
  root: data/pet                # Local single-label
  root: data/multi_label.csv    # Local multi-label
  root: username/dataset        # HuggingFace dataset
```

### Training Commands
```bash
# Single GPU
python main.py --cfgs configs/classification/pet.yaml

# Multi-GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 main.py --cfgs configs/classification/pet.yaml

# Optional arguments:
#   --sync_bn    # Sync BatchNorm layers
#   --resume     # Resume from checkpoint
#   --load_from  # Fine-tune from pretrained
```

Training logs will be saved to `run/exp/log{timestamp}.log`, containing dataset statistics, training progress, validation metrics and evaluation commands.
```bash
# View training log
vi run/exp/log{timestamp}.log  # e.g., log20241113-155144.log
```

## 📊 Evaluation

### Model Visualization
```bash
# Basic usage
python visualize.py --cfgs <config.yaml> \
                   --weight <best.pt> \
                   --class_json <class_indices.json> \
                   --data <val_data_path> \
                   --target_class <class_name> \
                   --ema

# Optional arguments:
#   --cam           # Enable CAM visualization
#   --badcase       # Group misclassified images
#   --sampling N    # Visualize N random samples
#   --remove_label  # Hide prediction text
```

Results will be saved to `visualization/exp/`:
- Single-label: Top-5 predictions, CAM heatmaps, badcase grouping
- Multi-label: Above-threshold predictions, per-class thresholds

### Model Validation
```bash
python validate.py --cfgs <config.yaml> \
                  --weight <best.pt> \
                  --eval_topk 5 \
                  --ema
```

## 🖼️ Example Results

<p align="center">
  <img src="../../misc/visual&validation.jpg" width="40%">
  <br>
  <em>Left: CAM visualization. Right: Validation metrics.</em>
</p>