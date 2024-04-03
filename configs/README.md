# Configuration instructions
I have prepared two configuration files for you in configs/[complete.yaml](./complete.yaml), [pet.yaml](./pet.yaml), you could modify based on your own task.

Please see below for the meaning of each parameter setting
* model
  ```
  model:
    # It can only start with the prefix "torchvision- or custom-" to indicate whether to use torchvision or a custom model.
    choice: torchvision-swin_t
  
    # Parameters that need to be passed for model initialization
    kwargs: {} 
    
    # number of classes, eg ImageNet 1000
    num_classes: 1000

    pretrained: True
  
    backbone_freeze: False
  
    bn_freeze: False
  
    bn_freeze_affine: False
  
    # transfer avg_pool to attention_pool
    attention_pool: False
  ```
* data
  ```
  data:
    # data path 
    root: data 
  
    # number of workers
    nw: 2 # dataloader中的num_workers
    # size of image
    imgsz: 
      - 480
      - 480
  
    # pipeline of data augment 
    train:
      # batchsize for one gpu, if 4 gpus, it means 4 x bs
      bs: 32 
      
      # Whether to make different augment pipelines for different classes. The default value is null.
      common_aug: null
      class_aug: null
        # a: 1 2 3
  
      # The augmentation method refers to utils/augment.py
      augment: 
        random_choice: 
        - random_color_jitter:
            prob: 0.5                 
            brightness: 0.1           
            contrast: 0.1             
            saturation: 0.1            
            hue: 0.1                  
        - random_cutout:
            n_holes: 4
            length: 80
            ratio: 0.2
            prob: 0.5
        random_crop_and_resize:
          size:
            - 480
            - 480
        # If no parameters need to be passed or default parameters are used, write "no_params"
        to_tensor: no_params 
        # If pre-training from ImageNet, add normalize. If from scratch, just delete it
        normalize: 
          mean: (0.485, 0.456, 0.406)
          std: (0.229, 0.224, 0.225)
  
      # set epochs for augmentation, all the augmentation will be dropped in remaining epochs(epochs-aug_epoch)
      aug_epoch: 40 
  ```
* hyp
  ```
  hyp:
    # total epochs
    epochs: 50

    lr0: 0.001
  
    lrf_ratio: null
  
    momentum: 0.937

    weight_decay: 0.0005
  
    warmup_momentum: 0.8
  
    # gradient warm-up for several epochs is not included in the total epochs and the learning rate increases linearly
    warm_ep: 3 
  
    # refer to # utils/loss.py
    loss: 
      ce: True           # Cross Entropy Loss
      bce:               # Binary Cross Entropy Loss
        - False # turn on or not
        - 0.5 # threshhold 
        - True # if multi_label

    label_smooth: 0.1

    strategy:
      # Progressive learning bundled with Mixup 
      prog_learn: False 
      mixup:
        - 0.1 # prob
        # Progressive learning and mixup stages, left closed and right open, gradient warm-up epoch excluded
        - [0,30]
  
      # FocalLoss only for BCELoss
      focal: 
        - False # turn on or not
        - 0.25 # alpha
        - 1.5 # gamma  
  
      # OHEM only for CELoss
      ohem: 
        - False # turn on or not
        - 8 # min_kept 
        - 0.7 # threshold for defining hard sample, a hard sample is that model probability greater than the threshold
        - 255 # ignore_index
    
    # refer to utils/optimizer.py
    optimizer: 
      - sgd # sgd adam or sam
      - False # Is there a special layer that needs to adjust the learning rate? -> built/layer_optimizer.py
  
    # learning rate decay, refer to utils/scheduler.py
    scheduler: cosine_with_warm
  ```