# Face Recognition Configuration instructions

1. **Attention** : Image input size has been fixed, ResNet and EfficientNet is 112x112, Swin Transformer is 224x224.  
2. The meaning of each parameter setting as below, more detail refer to [Classification Config README](../classification/README.md)  

* model
  * [Model Config](backbone_conf.yaml)
  * [Head Config](head_conf.yaml)  
    
  ```markdown
   model:
     task: face
     backbone:
       resnet:
           # only {50, 100, 152} is supported
           depth: 50
           drop_ratio: 0.4
           # only {ir, ir_se} is supported
           net_mode: ir_se
           feat_dim: 512
           # [out_h, out_w] decide the last linear layer, see 127-Line in backbone/ResNets.py
           out_h: 7
           out_w: 7
     head:
       arcface:
         feat_dim: 512
         num_class: 9628
         margin_arc: 0.35
         margin_am: 0.0
         scale: 32
    ```

* data
```markdown
data:
  root: ms_celeb  # -train -val
  nw: 4 # if not multi-nw, set to 0
  train:
    bs: 64 # one gpus if DDP
    common_aug: null 
    class_aug: null
      #S: 0 1 2 4 5 6
      #B-: 0 1 2 4 5 6
    augment: # refer to utils/augment.py
      random_cutout:
        n_holes: 2
        length: 15
        prob: 0.5
      random_grayscale:
        p: 0.5
      random_choice:
        - random_localgaussian:
            ksize:
              - 17
              - 17
        - random_gaussianblur:
            prob: 0.5
      random_horizonflip:
        p: 0.5
      pad2square: no_params
      resize:
        size: 112
      to_tensor: no_params
      normalize: # default use ImageNet1K mean & var, if not pretrained, del normalize 
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
    aug_epoch: 28 # augment for epochs, on which epoch to weaken, except warm_epoch
  val:
    bs: 32
    pair_txt: ms_celeb/pair.txt
    augment:
        pad2square: no_params
        resize:
          size: 112
        to_tensor: no_params
        normalize: # default use ImageNet1K mean & var, if not pretrained, del normalize
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
```
* hyp
```markdown
hyp:
  epochs: 30
  lr0: 0.01 # sgd=1e-2, adam=1e-3
  lrf_ratio: null # decay to lrf_ratio * lr0, if None, 0.1
  momentum: 0.937
  weight_decay: 0.0005
  warmup_momentum: 0.8
  warm_ep: 2
  loss:
    ce: True
  label_smooth: 0.0
  optimizer: 
    - sgd # sgd, adam or sam
    - False # Different layers in the model set different learning rates, in built/layer_optimizer
  scheduler: cosine_with_warm # linear or cosine
```