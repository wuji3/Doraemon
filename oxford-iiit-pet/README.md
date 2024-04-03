## Oxford-IIIT Pet  
### Introduction 
This is a pet dataset with 37 categories and approximately 200 images per category. The images vary greatly in proportion, pose and lighting. All images have associated ground truth annotations for breed, head ROI, and pixel-level tripartite segmentation  
1. Paper: http://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/parkhi12a.pdf
2. URL: https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz 

### Data Prepare  
<details close>
<summary>Tips 🌟</summary>

1. Remember to install the environment [Install](../README.md)
2. If your network is limited and you cannot download through the URL, I have prepared Baidu Cloud for you.  
    Link：https://pan.baidu.com/s/1PjM6kPoTyzNYPZkpmDoC6A   
    Code：yjsl 
</details>
 
<details close>
<summary>Run Script  🚀️</summary>

Unzip oxford-iiit-pet.tgz to the path as followed. Then, start split2dataset.py. The directory structure will look like this. There will be an extra pet folder with train and val divided into it.  

```shell
cd oxford-iiit-pet
python split2dataset.py
```

```
project                    
│
├── oxford-iiit-pet  
│   ├── oxford-iiit-pet   (directory after zipping)
│       ├── annotations
│       ├── images
├── split2dataset.py

          |
          |
         \|/   
         
project                    
│
├── oxford-iiit-pet  
│   ├── oxford-iiit-pet
│       ├── annotations
│       ├── images
│   ├── pet   (after start split2dataset.py)
│       ├── train
│       ├── val
├── split2dataset.py
```
```
--vision(根目录)
    --oxford-iiit-pet
        --oxford-iiit-pet
            --annotations
            --images
        --pet(脚本执行后多出的文件夹)
            --train
            --val
        --split2dataset.py

```
</details>

### Train & Infer
Refer to [README.md](../README.md)