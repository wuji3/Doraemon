## <div align="center">Image Classification</div>

## Tutorials
<details open>
<summary>Data</summary>

1. [Oxford IIIT Pet](../../oxford-iiit-pet/README.md) : A pet dataset with 37 categories and approximately 200 images per category. 
2. Custom Data: Prepare your data as below.
3. If you only have multiple images in some category-folder, the script will be useful.
- ```python
  python tools/data_prepare.py --postfix <jpg | png> --root <input your data realpath> --frac <segment ratio of train-set per category, eg: 0.9 0.6 0.3 0.9 0.9>
  ```

</details>

<details open>
<summary>Configuration ️</summary>

**ATTENTION**, [Config Instructions](../../configs/classification/README.md) Is All You Need 🌟
- [_Oxford IIIT Pet_] [pet.yaml](../../configs/classification/pet.yaml) has prepared for you.
- [_Custom Data_]  Modify based on [config.yaml](../../configs/classification/complete.yaml) or [pet.yaml](../../configs/classification/pet.yaml)  

</details>

<details open>
<summary>Training 🚀️️</summary>

```shell
# one machine one gpu
python main.py --cfgs configs/classification/pet.yaml

# one machine multiple gpus
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 main.py --cfgs configs/classification/pet.yaml
                                                                 --sync_bn[Option: this will lead to training slowly]
                                                                 --resume[Option: training from checkpoint]
                                                                 --load_from[Option: training from fine-tuning]
```
</details>

<details open>
<summary>Validate & Visualization 🌟🌟</summary>

```markdown
# You will find context below in log when training completes.

Training complete (0.093 hours)  
Results saved to /home/duke/project/vision-face/run/exp3  
Predict:         python visualize.py --cfgs /xxx/.../vision-classifier/run/exp/pet.yaml --weight /xxx/.../vision-classifier/run/exp/best.pt --badcase --class_json /xxx/.../vision-classifier/run/exp/class_indices.json --ema --cam --data <your data>/val/XXX_cls 
Validate:        python validate.py --cfgs /xxx/.../vision-classifier/run/exp/pet.yaml --eval_topk 5 --weight /xxx/.../vision-classifier/run/exp/best.pt --ema
```

```shell
# visualize.py provides the attention heatmalp function, which can be called by passing "--cam"
python visualize.py --cfgs /xxx/.../vision-classifier/run/exp/pet.yaml --weight /xxx/.../vision-classifier/run/exp/best.pt  --class_json /xxx/.../vision-classifier/run/exp/class_indices.json --data <your data>/val/XXX_cls
                                                                                                                                                                                               --ema[Option: may improve performance a bit] 
                                                                                                                                                                                               --cam[Option: show the attention heatmap]
                                                                                                                                                                                               --badcase[Option: group the badcase in a folder]
                                                                                                                                                                                               --target_class[Option: which catogary do you want to check, serving for --badcase, if not set, directory-name will be regarded as catogary]
                                                                                                                                                                                               --no_annotation[Optition: remove model description on left-top]
```

```shell
python validate.py --cfgs /xxx/.../vision-classifier/run/exp/pet.yaml --eval_topk 5 --weight /xxx/.../vision-classifier/run/exp/best.pt 
                                                                                    --ema[Option: may improve performance a bit]
```

The picture below is the visualize and validate result. It is for visual reference only.

<p align="center">
  <img src="../../misc/visual&validation.jpg" width="40%" height="auto" >
</p>
</details>
