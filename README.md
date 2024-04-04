## <div align="center">vision classifier</div>
[中文](./README_ch.md)
## Tutorials

<details open>
<summary>Install ☘️</summary>

```shell
# It is recommanded to create a separate virtual environment
conda create -n vision python=3.9 
conda activate vision

# torch==2.0.1(lower is also ok) -> https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio cpuonly -c pytorch # cpu-version
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia  # cuda-version

pip install -r requirements.txt

# Without Arial.ttf, inference may be slow due to network IO.
mkdir -p ~/.config/Ultralytics
cp misc/Arial.ttf ~/.config/Ultralytics
```
</details>

<details close>
<summary>Data 🚀️</summary>

[If for learning, refer to oxford-iiit-pet](./oxford-iiit-pet/README.md)
```bash
python tools/data_prepare.py --postfix <jpg or png> --root <input your data realpath> --frac <train segment ratio, eg: 0.9 0.6 0.3 0.9 0.9>
```

```markdown
project                    
│
├── data  
│   ├── clsXXX-1   
│   ├── clsXXX-... 
├── tools
│   ├── data_prepare.py  

          |
          |
         \|/   
     
project
│
├── data  
│   ├── train  
│       ├── clsXXX 
│           ├── XXX.jpg/png 
│   ├── val  
│       ├── clsXXX 
│           ├── XXX.jpg/png 
├── tools
│   ├── data_prepare.py  
```

</details>

<details close>
<summary>Configuration 🌟🌟️</summary>

If custom data, refer to [Config](./configs/README.md) for writing your own config.  (Recommend🌟: modify based on [complete.yaml](./configs/complete.yaml) or [pet.yaml](./configs/pet.yaml))  
If [oxford-iiit-pet](./oxford-iiit-pet/README_ch_.md), [pet.yaml](./configs/pet.yaml) has prepared for you.
</details>

<details close>
<summary>Training 🌟️</summary>

```shell
# one machine one gpu
python main.py --cfgs configs/pet.yaml

# one machine multiple gpus
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 main.py --cfgs configs/pet.yaml
```
</details>

<details close>
<summary>Validate & Visualization 🌟🌟</summary>

<p align="center">
  <img src="./misc/visual&validation.jpg" width="40%" height="auto" >
</p>

```markdown
# You will find context below in log when training completes.

Training complete (0.093 hours)  
Results saved to /home/duke/project/vision-face/run/exp3  
Predict:         python visualize.py --cfgs /xxx/.../vision-classifier/run/exp/pet.yaml --weight /xxx/.../vision-classifier/run/exp/best.pt --badcase --class_json /xxx/.../vision-classifier/run/exp/class_indices.json --ema --cam --data <your data>/val/XXX_cls 
Validate:        python validate.py --cfgs /xxx/.../vision-classifier/run/exp/pet.yaml --eval_topk 5 --weight /xxx/.../vision-classifier/run/exp/best.pt --ema
```

```shell
# visualize.py provides the attention heatmalp function, which can be called by passing "--cam"
python visualize.py --cfgs /xxx/.../vision-classifier/run/exp/pet.yaml --weight /xxx/.../vision-classifier/run/exp/best.pt --badcase --class_json /xxx/.../vision-classifier/run/exp/class_indices.json --ema --cam --data <your data>/val/XXX_cls
```
```shell
python validate.py --cfgs /xxx/.../vision-classifier/run/exp/pet.yaml --eval_topk 5 --weight /xxx/.../vision-classifier/run/exp/best.pt --ema
```

</details>

## Method & Paper
| Method                                                   | Paper                                                                            |
|----------------------------------------------------------|----------------------------------------------------------------------------------|
| [SAM](https://arxiv.org/abs/2010.01412v3)                | Sharpness-Aware Minimization for Efficiently Improving Generalization            |
| [Progressive Learning](https://arxiv.org/abs/2104.00298) | EfficientNetV2: Smaller Models and Faster Training                               |
| [OHEM](https://arxiv.org/abs/1604.03540)                 | Training Region-based Object Detectors with Online Hard Example Mining           |
| [Focal Loss](https://arxiv.org/abs/1708.02002)           | Focal Loss for Dense Object Detection                                            |
| [Cosine Annealing](https://arxiv.org/abs/1608.03983)     | SGDR: Stochastic Gradient Descent with Warm Restarts                             |
| [Label Smoothing](https://arxiv.org/abs/1512.00567)      | Rethinking the Inception Architecture for Computer Vision                        |
| [Mixup](https://arxiv.org/abs/1710.09412)                | MixUp: Beyond Empirical Risk Minimization                                        |
| [CutOut](https://arxiv.org/abs/1708.04552)               | Improved Regularization of Convolutional Neural Networks with Cutout             |
| [Attention Pool](https://arxiv.org/abs/2112.13692)       | Augmenting Convolutional networks with attention-based aggregation               |
| [GradCAM](https://arxiv.org/abs/1610.02391)              | Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization |

## Model & Paper

| Method                                                 | Paper                                                                 | Name in configs, eg: torchvision-mobilenet_v2                                   |
|--------------------------------------------------------|-----------------------------------------------------------------------|---------------------------------------------------------------------------------|
| [MobileNetv2](https://arxiv.org/abs/1801.04381)        | MobileNetV2: Inverted Residuals and Linear Bottlenecks           | mobilenet_v2                                                                    |
| [MobileNetv3](https://arxiv.org/abs/1905.02244)        | Searching for MobileNetV3                     | mobilenet_v3_small, mobilenet_v3_large                                          |
| [ShuffleNetv2](https://arxiv.org/abs/1807.11164)       | ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design | shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0  |
| [ResNet](https://arxiv.org/abs/1512.03385)             | Deep Residual Learning for Image Recognition                                 | resnet18, resnet34, resnet50, resnet101, resnet152                              |
| [ResNeXt](https://arxiv.org/abs/1611.05431)            | Aggregated Residual Transformations for Deep Neural Networks                  | resnext50_32x4d, resnext101_32x8d, resnext101_64x4d                             |
| [ConvNext](https://arxiv.org/abs/2201.03545)           | A ConvNet for the 2020s             | convnext_tiny, convnext_small, convnext_base, convnext_large                    |
| [EfficientNet](https://arxiv.org/abs/1905.11946)       | EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks                             | efficientnet_b{0..7}                                          |
| [EfficientNetv2](https://arxiv.org/abs/2104.00298)     | EfficientNetV2: Smaller Models and Faster Training  | efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l            |
| [Swin Transformer](https://arxiv.org/abs/2103.14030)   | Swin Transformer: Hierarchical Vision Transformer using Shifted Windows    | swin_t, swin_s, swin_b              |
| [Swin Transformerv2](https://arxiv.org/abs/2111.09883) | Swin Transformer V2: Scaling Up Capacity and Resolution | swin_v2_t, swin_v2_s, swin_v2_b |


## Tools  
1. Split the data set into training set and validation set
```shell
python tools/data_prepare.py --postfix <jpg or png> --root <input your data realpath> --frac <train segment ratio, eg: 0.9 0.6 0.3 0.9 0.9>
```
2. Data augmented visualization 
```shell
python tools/test_augment.py
```
![](misc/aug_image.png)