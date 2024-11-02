# Oxford-IIIT Pet Dataset Preparation

## Introduction
The Oxford-IIIT Pet Dataset contains images of 37 pet breeds, with approximately 200 images per category. The dataset features variations in scale, pose, and lighting. Each image includes annotations for breed, head ROI, and pixel-level tripartite segmentation.

- **Paper**: [Parkhi et al. (2012)](http://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/parkhi12a.pdf)
- **Dataset URL**: [Oxford-IIIT Pet Dataset](https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz)

## Data Preparation

### Prerequisites
- Ensure you have installed the required environment. See [Installation Guide](../README.md) for details.
- If you have trouble downloading from the provided URL, use our Baidu Cloud link:
  - Link: https://pan.baidu.com/s/1PjM6kPoTyzNYPZkpmDoC6A
  - Code: yjsl

### Steps

1. Download and extract the `oxford-iiit-pet.tgz` file to the `data` directory.

2. Run the data preparation script:
   ```shell
   cd data
   python split2dataset.py
   ```

3. The script will organize the data and clean up temporary files. After execution, your `data` directory will contain:
   - `pet/` directory with `train/` and `val/` subdirectories
   - `split2dataset.py` script

### Directory Structure
```
Before:                 After:
data/                   data/
├── oxford-iiit-pet/    ├── pet/
│   ├── annotations/    │   ├── train/
│   └── images/         │   │   ├── Abyssinian/
├── split2dataset.py    │   │   ├── Beagle/
                        │   │   └── ...
                        │   └── val/
                        │       ├── Abyssinian/
                        │       ├── Beagle/
                        │       └── ...
                        └── split2dataset.py
```

## Training & Inference
For instructions on how to train models and perform inference using this dataset, please refer to the [main README](../models/classifier/README.md).