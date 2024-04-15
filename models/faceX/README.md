## <div align="center">Face Recognition</div>

## Tutorials
<details open>
<summary>Data</summary>

1. Training Data: We use [MS-Celeb-1M-v1c](http://trillionpairs.deepglint.com/data) for conventional training. To remove the identities which may overlap between this dataset, a cleaner id-list can be found in [CLEAN](https://github.com/IvyHuang-25/CleanAndRelabel-MS-Celeb-1M). After washing, the dataset contains 79077 identities, over 367 million faces.
2. Testing Data: We provide [LFW](https://pan.baidu.com/s/1y4UXQkjv5PnY_6CTV_K2xQ), extracted code is **yjsl**. Including data and pairs.txt.

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
</details>

## Experiment
| Backbone | MS CELEB |    Device     | Period         | LFW w/o EMA |
|:--------:|:--------:|:-------------:|:---------------|:--------------:| 
| ResNet50 | 10000 ID | RTX2080Ti x 2 | 30Epoch/600min | 98.01%/98.21%  |
