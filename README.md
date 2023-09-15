# vision_classification
Provide baseline to solve different kinds of vision classification tasks   

<font color=Red>in some days, repo will add kinds of Attentions-Modules, so stay tuned</font>
## introduce
  params setting refer to configs/complete.yaml, include model, data, hyp 
    
  model: support BN freeze, backbone freeze. set out_channels as you like, maybe not equal to num_classes if you want to do training strategy  
  data: support custom-style data augment, all the data augment are placed in utils/augment.py, eg: cutout, colerjitter...  
  hyp: support many loss function eg: ce and bce, support custom-style training stategy, eg: focalloss, mixup and progressive learning  
  training: support ema, progressive learning, focalloss, mixup ... 
    
## usage
  1、the repo support "split epochs into augment-epoch and no-augment-epoch, set in XXX.yaml" -> aug_epoch: 25, means augment for 25 epochs then train in no-augment  
  2、the repo support "OHEM, set in XXX.yaml" -> bceloss: True, focal: True 30, True means turn on focalloss, 30 means bceloss before 30th epoch and focalloss after. if no focalloss, just set focal: 0 30  
  3、the repo support "mixup in given epochs, set in XXX.yaml" -> mixup: 0.2 [20,30], 0.2 means prob to use mixup every epoch, [20,30] means interval to start/delete mixup. if no mixup, just set mixup: 0 [20,30]  
  4、the repo support "progressive learning $refer to EfficientV2, set in XXX.yaml" -> prog_learn: True, will effect on image size & mixup, divide into 3 parts in default 

## support models: torchvision-xxx
    mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
    shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
    resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, resnext101_64x4d
    convnext_tiny, convnext_small, convnext_base, convnext_large
    efficientnet_b0 -> efficientnet_b7, efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l

## prepare the data
    --data
      --train
        -- clsXXX
          -- XXX.jpg/png
      --val
        -- clsXXX
          -- XXX.jpg/png
## single GPU
    python main.py
## DDP
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 main.py
