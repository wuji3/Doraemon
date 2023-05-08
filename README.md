# vision_classification
Provide baseline to solve different kinds of vision classification tasks
## introduce
  params setting refer to configs/complete.yaml, include model, data, hyp 
    
  model: support BN freeze, backbone freeze. set out_channels as you like, maybe not equal to num_classes if you want to do training strategy  
  data: support custom-style data augment, all the data augment are placed in utils/augment.py, eg: cutout, colerjitter...  
  hyp: support many loss function eg: ce and bce, support custom-style training stategy, eg: focalloss and mixup  
    
## usage
  1、the repo support "split epochs into augment-epoch and no-augment-epoch, set in XXX.yaml" -> aug_epoch: 25, means augment for 25 epochs then train in no-augment  
  2、the repo support "OHEM, set in XXX.yaml" -> bceloss: True, focal: 1 30, 1 means turn on focalloss, 30 means bceloss before 30th epoch and focalloss after. if no focalloss, just set focal: 0 30  
  3、the repo support "mixup in given epochs, set in XXX.yaml" -> mixup: 0.2 25, 0.2 means prob to use mixup every epoch, 25 means mixup effect for 25 epochs. if no mixup, just set mixup: 0 25  

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
