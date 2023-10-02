# vision classifier
本仓库提供baseline用于图像分类任务，会持续维护，方便大家用于项目和竞赛，支持以下trick和module:

`SAM                  | Sharpness-Aware Minimization for Efficiently Improving Generalization | https://arxiv.org/abs/2010.01412v3`  
`Progressive Learning | EfficientNetV2: Smaller Models and Faster Training                    | https://arxiv.org/abs/2104.00298`  
`OHEM                 | Training Region-based Object Detectors with Online Hard Example Mining| https://arxiv.org/abs/1604.03540`  
`Focal Loss           | Focal Loss for Dense Object Detection                                 | https://arxiv.org/abs/1708.02002`  
`CosineAnnealing      | SGDR: Stochastic Gradient Descent with Warm Restarts                  | https://arxiv.org/abs/1608.03983`  
`Label Smoothing      | Rethinking the Inception Architecture for Computer Vision             | https://arxiv.org/abs/1512.00567`  
`Mixup                | MixUp: Beyond Empirical Risk Minimization                             | https://arxiv.org/abs/1710.09412`  
`CutOut               | Improved Regularization of Convolutional Neural Networks with Cutout  | https://arxiv.org/abs/1708.04552`   

另外还有一些比赛通用trick，如EMA(Exponential Moving Average)，MLT(Multi Label Training)，AMP(Automatic Mixed Precision)等，若读者找到相应论文可以私信我加上

## 仓库简介
* 训练可视化参考YOLOv5 -> https://github.com/ultralytics/yolov5
* 本仓库解耦Dataset, Augment Pipeline, Optimizer, Scheduler, Sampler, Loss, Engine, Model，现在Model是使用Torchvision的模型，如果想用自己的模型，需要稍稍改动代码，这一块作者没需求所以一直空着没写 
* 所有的策略只修改configs/complete.yaml这个配置文件就可以完成了，方便大家写论文跑实验、打比赛和做项目
#### 配置文件简介
* model
  ```
  model:
    # 只能以”torchvision-或custom-“前缀开头 表示使用torchvision还是自定义的模型
    choice: torchvision-swin_t
  
    # 模型初始化需要传的参数 torchvision一般都有默认配置 除非你有更改的需求 比如Dropout 若改模型结构相关的参数 则预训练参数就不能用了
    kwargs: {} 
    
    # 输出类别
    num_classes: 1000 # out_channels of fc
  
    # 是否载入ImageNet预训练参数
    pretrained: True
  
    # 是否冻结backbone 不包括最后的全连接层 该功能支持模型请看下面章节 [当前支持的模型: torchvision-xxx]
    backbone_freeze: False
  
    # 是否冻结batchnorm的统计量 即m.eval()
    bn_freeze: False
  
    # 是否冻结batchnorm的beta和gamma 即requires_grad=False
    bn_freeze_affine: False
  ```
* data
  ```
  data:
    # 数据路径
    root: ./data 

    nw: 2 # dataloader中的num_workers
    imgsz: # 训练尺寸 (h, w) 和 augment最后一个resize保持一致
      - 480
      - 480
  
    # 训练集pipeline
    train:
      bs: 32 # 单卡的batchsize 若4卡训练batchsize等价于32x4=128
      
      # 是否对不同类别做不同的增强pipeline 默认写null
      common_aug: null
      class_aug: null
        # a: 1 2 3
  
      # 训练集数据增强baseline 增强手段在utils/augment.py
      augment: 
        random_choice: # 从random_color_jitter和random_cutout中随机挑选一个
        - random_color_jitter: # 增强方法
            prob: 0.5                 #-传
            brightness: 0.1           #-相
            contrast: 0.1             #-关
            saturation: 0.1           #-参 
            hue: 0.1                  #-数
        - random_cutout:
            n_holes: 4
            length: 80
            ratio: 0.2
            prob: 0.5
        random_crop_and_resize:
          size:
            - 480
            - 480
        to_tensor: no_params # 若没有参数需要传或用默认参数 则写no_params
        normalize: # 使用ImageNet预训练参数 则加normalize 若从头训 可以删掉这个配置
          mean: (0.485, 0.456, 0.406)
          std: (0.229, 0.224, 0.225)
  
      # 多少个epoch启用数据增强 可以等于hyp中的epochs 等价于全程增强 或者3/4epochs增强 留最后1/4epochs取消增强 回归真实分布
      aug_epoch: 40 
  ```
* hyp
  ```
  hyp:
    # 一共训练多少个epoch 注意和data中aug_epoch的联系
    epochs: 50
  
    # 启动学习率
    lr0: 0.001
  
    # 收尾学习率因子α 即学习率衰减至lr0 x α，null则默认α=0.1 
    lrf_ratio: null
  
    # 动量 参考yolov5
    momentum: 0.937

    # L2
    weight_decay: 0.0005
  
    # 梯度热身期动量 参考yolov5
    warmup_momentum: 0.8
  
    # 梯度热身几个epoch 不计入总epoch 学习率线性递增
    warm_ep: 3 
  
    # 监督方式
    loss: # utils/loss.py
      ce: True           # Cross Entropy Loss
      bce:               # Binary Cross Entropy Loss
        - False # 是否开启
        - 0.5 # 阈值 
        - True # multi_label 是否开启multi-label
  
    # 标签平滑
    label_smooth: 0.1
  
    # 训练策略
    strategy:
      # 渐进式学习与Mixup绑定 渐进共3个阶段 imgsz_ratio 0.5 -> 0.75 -> 1，mixup β分布中α 0 -> 0.1 -> 0.2, 
      # eg: imgsz=480 epochs=100 warm_ep=3 aug_epoch=80 mixup阶段=[0, 70] 
      # 前3个热身epoch不开启数据增强 模型热身
      # 0-35epoch 240尺寸训练 mixup β分布中α=0 + 数据增强
      # 35-70epoch 360尺寸训练 mixup β分布中α=0.1 + 数据增强
      # 70-80epoch 480尺寸训练 mixup β分布中α=0.2 + 数据增强
      # 80-100epoch 480尺寸训练 mixup和数据增强全部关闭 
      prog_learn: True 
      mixup:
        - 0.1 # prob
        - [0,30] # 渐进式学习和mixup作用阶段 左闭右开 梯度热身epoch排除在外
  
      # FocalLoss 仅支持BCELoss
      focal: 
        - False # 是否开启
        - 0 #  从第几个epoch开启focalloss 排除warmup 即梯度热身期间是正常bce
  
      # OHEM 困难样例挖掘 仅支持CELoss
      ohem: 
        - False # 是否开启
        - 8 # min_kept 最小采样数
        - 0.7 # 概率阈值 即低于该阈值为困难样例
        - 255 # ignore_index 不要改动这个参数除非你有需要忽略的类别
    
    # 优化器 utils/optimizer.py
    optimizer: 
      - sgd # sgd adam or sam
      - [null, 0.1] # [layer, alpha] 特殊的层需要调学习率 built/layer_optimizer.py
  
    # 学习率衰减策略 utils/scheduler.py
    scheduler: cosine_with_warm
  ```
## 当前支持的模型: torchvision-xxx
    mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
    shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
    resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, resnext101_64x4d
    convnext_tiny, convnext_small, convnext_base, convnext_large
    efficientnet_b0 -> efficientnet_b7, efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l
    swin_t, swin_s, swin_b, swin_v2_t, swin_v2_s, swin_v2_b

## 使用指南
1. 环境准备  
```shell
# torch==2.0.1(低几个版本也可以) -> https://pytorch.org/get-started/locally/
conda create -n vision python=3.9 # 建议单独创建一个虚拟环境
conda activate vision
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia # 安装torch
pip install -r requirements.txt
```
2. 数据准备
```
--vision
  --data
    --train
      -- clsXXX
        -- XXX.jpg/png
    --val
      -- clsXXX
        -- XXX.jpg/png
```
3. 启动训练
```shell
# 单机单卡
python main.py

# 单机多卡
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 main.py
```
4. 启动验证和推理
```
训练结束后，脚本会自动提示验证和推理的命令 predict.py中传入--badcase会自动把模型预测错误的样本存在badcase文件夹中 方便大家分析样本
python val.py --weight run/exp9/best.pt --choice torchvision-swin_t --kwargs "{}" --root data --num_classes 5 --transforms "{'resize': {'size': [640, 640]}, 'to_tensor': 'no_params', 'normalize': {'mean': '(0.485, 0.456, 0.406)', 'std': '(0.229, 0.224, 0.225)'}}" --thresh 0 --head ce --multi_label False
python predict.py --weight run/exp9/best.pt --badcase --save_txt --choice torchvision-swin_t --kwargs "{}" --class_head ce --class_json run/exp9/class_indices.json --num_classes 5 --transforms "{'resize': {'size': [640, 640]}, 'to_tensor': 'no_params', 'normalize': {'mean': '(0.485, 0.456, 0.406)', 'std': '(0.229, 0.224, 0.225)'}}" --root data/val/XXX_cls
```
5. 界面简介  
CELoss训练指标参考ImageNet 有Top1和Top5
![](./configs/celoss.jpg)
BCELoss训练指标有Precision Recall F1-Score
![](./configs/bceloss.jpg)

## 其他
1. 数据增强可视化调试  
仓库提供数据增强可视化脚本，在做策略之前，可以启动test_augment.py看一下想用的增强方式 适不适用于当前数据
