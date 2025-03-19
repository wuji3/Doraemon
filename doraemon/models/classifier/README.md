# <div align="center">Image Classification</div>                                                                                                                                                          
 
## 📦 Data Preparation
 
### Quick Start with Pre-prepared Datasets
1. **Using HuggingFace Dataset (Recommended)**
   
   Dataset: [wuji3/oxford-iiit-pet](https://huggingface.co/datasets/wuji3/oxford-iiit-pet)
   ```yaml
   # In your config file (e.g., configs/classification/pet.yaml)
   data:
     root: wuji3/oxford-iiit-pet
   ```
 
2. **Download Pre-prepared Dataset**
   - Oxford-IIIT Pet Dataset (37 pet breeds)
     - [Download from Baidu Cloud](https://pan.baidu.com/s/1PjM6kPoTyzNYPZkpmDoC6A) (Code: yjsl) **Recommended**
     - [Download from Official URL](https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz)
   ```bash
   # After downloading:
   cd data
   tar -xf oxford-iiit-pet.tgz
   python split2dataset.py
   ```
 
### Training with Your Own Dataset
You can prepare your data in either single-label or multi-label format:
 
#### Option 1: Single-label Format
```
your_dataset/
├── train/
│   ├── class1/             # Folder name = class name
│   │   ├── image1.jpg
│   │   └── ...
│   └── class2/
│       └── ...
└── val/
    ├── class1/
    │   └── ...
    └── class2/
        └── ...
```
 
#### Option 2: Multi-label Format (CSV)
Create a CSV file with the following structure:
```csv
image_path,tag1,tag2,tag3,train
/path/to/image1.jpg,1,0,1,True    # 1=has_tag, 0=no_tag
/path/to/image2.jpg,0,1,0,True    # True=training set
```
 
### Data Preparation Helper
Convert a folder of categorized images into the required training format:
```bash
# If your data structure is:
# your_dataset/
# ├── class1/
# │   ├── img1.jpg
# │   └── img2.jpg
# ├── class2/
# │   ├── img3.jpg
# │   └── img4.jpg
# └── ...

python tools/data_prepare.py \
    --root path/to/your/images \
    --postfix jpg \          # Image format: jpg or png
    --frac 0.8              # Split ratio: 80% training, 20% validation
```
 
This script will automatically:
1. Create train/ and val/ directories
2. Split images from each class into training and validation sets
3. Maintain the class folder structure in both sets
 
## 🧊 Models
 
### Model Configuration
```yaml
model:
  task: classification
  name: timm-swin_base_patch4_window7_224  # Format: timm-{model_name}
  image_size: 224
  num_classes: 35
  pretrained: True
  kwargs: {}  # Additional parameters for model initialization
```
 
### Available Models
```python
import timm
timm.list_models(pretrained=True)  # ['beit_base_patch16_224.in22k_ft_in22k', 'swin_base_patch4_window7_224.ms_in22k_ft_in1k', 'vit_base_patch16_siglip_224.webli', ...]
```

## 🚀 Training
 
### Training Options

```bash
# Single GPU training
python -m scripts.train configs/recognition/pet.yaml

# Multi-GPU training
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 -m scripts.train configs/recognition/pet.yaml [options]
```

**Options:**
- `--resume path/to/model.pt`: Resume interrupted training
- `--sync_bn`: Enable synchronized BatchNorm for multi-GPU

### Monitor Training
```bash
# View real-time training log
tail -f scripts/run/exp/log{timestamp}.log  # e.g., log20241113-155144.log
```
 
## 📊 Evaluation & Visualization
 
### Analyze Model Predictions

```bash
python -m scripts.infer scripts/run/exp/best.pt --data <path/to/dataset> [options]
```

**Options:**
- `--infer-option {default, autolabel}`: 
  - `default`: Infer + Visualize + CaseAnalysis
  - `autolabel`: Infer + Label
- `--classes A B C`: Filter specific classes
- `--split val`: Only analyze validation set
- `--sampling 10`: Analyze a random subset of samples
- `--ema`: Use Exponential Moving Average for model weight

### Validate Model Performance
```bash
python -m scripts.validate scripts/run/exp/best.pt --ema
```
 
## 🖼️ Example

<p align="center">
  <img src="../../../misc/training.jpg" height="200px" style="display: inline-block; vertical-align: top;">
  <img src="../../../misc/eval_class.jpeg" height="200px" style="display: inline-block; vertical-align: top;">
  <img src="../../../misc/gradcam.jpg" height="200px" style="display: inline-block; vertical-align: top;">
  <br>
  <em>Visualization of Model Processes (Left: Training, Center: Evaluation, Right: Inference)</em>
</p>                                              
