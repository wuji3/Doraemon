## <div align="center">Face Recognition</div>

## Tutorials
<details open>
<summary>Data</summary>

1. Training Data: We use [MS-Celeb-1M-v1c](http://trillionpairs.deepglint.com/data) for conventional training. To remove the identities which may overlap between this dataset, a cleaner id-list can be found in [CLEAN](https://github.com/IvyHuang-25/CleanAndRelabel-MS-Celeb-1M). After washing, the dataset contains 79077 identities, over 367 million faces.
2. Testing Data: We provide [LFW](https://pan.baidu.com/s/1y4UXQkjv5PnY_6CTV_K2xQ), extracted code is **yjsl**. Including data and pairs.txt.
3. Testing Data For Face Evaluation Only Support 5000~6000 Pairs Now.
```markdone
facedata/
├── pairs.txt
├── train
│   └── 11272
│       ├── 0-FaceId-0.jpg
│       └── 1-FaceId-0.jpg
└── val
    ├── Micky_Ward
    │   └── Micky_Ward_0001.jpg
    └── Miguel_Aldana_Ibarra
        └── Miguel_Aldana_Ibarra_0001.jpg
```
</details>

<details open>
<summary>Configuration ️</summary>

**ATTENTION**, [Config Instructions](../../configs/faceX/README.md) Is All You Need 🌟
- [_MS CELEB_] [face.yaml](../../configs/faceX/face.yaml) has prepared for you.
- [_Custom Data_]  Modify based on [face.yaml](../../configs/faceX/face.yaml)

</details>

<details open>
<summary>Training 🚀️️</summary>

```shell
# one machine one gpu
python main.py --cfgs configs/faceX/face.yaml

# one machine multiple gpus
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 main.py --cfgs configs/faceX/face.yaml --print_freq 50 --save_freq 5
                                                                 --sync_bn[Option: this will lead to training slowly]
                                                                 --resume[Option: training from checkpoint]
                                                                 --load_from[Option: training from fine-tuning]
```

</details>

<details open>
<summary>Validate & Visualization 🌟</summary>

```markdown
# You will find context below in log when training completes.

Training complete (10.652 hours)                                                                                                                                                                                                                   
Results saved to /root/xxx/vision/run/exp
Validate:        python validate.py --cfgs configs/faceX/face.yaml --weight /root/xxx/vision/run/exp/which_weight --ema 
```

```shell
python validate.py --cfgs configs/faceX/face.yaml --weight /root/xxx/vision/run/exp/which_weight 
                                                  --ema[Option: may improve performance a bit] 
```

```shell
# You may want to observe some trends, such as Train_loss, Train_lr, Val_mean, Val_std
tensorboard --logdir /root/xxx/vision/run/exp
```

The picture below is the training result using 563 identities(27972 images, 32 epochs). It is for visual reference only.
<p align="center">
  <img src="../../misc/tensorboard.jpg" width="70%" height="auto" >
</p>
</details>

## Experiment
| Backbone | MS CELEB |    Device     | Period         | LFW w/o EMA |
|:--------:|:--------:|:-------------:|:---------------|:--------------:| 
| ResNet50 | 10000 ID | RTX2080Ti x 2 | 30Epoch/600min | 98.01%/98.21%  |
