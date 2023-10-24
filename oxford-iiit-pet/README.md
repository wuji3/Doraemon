## Oxford-IIIT Pet 实战  
### 数据集简介  
这是一个有37个类别的宠物数据集，每个类别大约有200张图像。这些图像在比例、姿势和灯光方面有很大的变化。所有图像都有相关的品种、头部 ROI 和像素级三图分割的地面实况注释。
1. 论文 >> http://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/parkhi12a.pdf
2. 下载地址 >> https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz 【科学上网】

### 数据准备  
1. 不要忘记完成[Vision](../README.md)的[使用指南] >> [1. 环境准备] 先把环境搭建好
2. 大部分同学可能没梯子 所以我上传到百度云盘了 链接：https://pan.baidu.com/s/1PjM6kPoTyzNYPZkpmDoC6A 提取码：yjsl (一键三连 嘿嘿)
3. 解压到oxford-iiit-pet路径下 目录结构长这个样子  
```
--vision(根目录)
    --oxford-iiit-pet
        --oxford-iiit-pet(解压后的文件夹)
            --annotations
            --images
        --split2dataset.py
```
4. 启动split2dataset.py 完成后 目录结构长这个样子 多出一个pet文件夹 里面切分好了 train和val
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
5. 说明:
切分完成后只有35个类别 数据集简介中说是有37个类别 这个大家不用纠结 我是按照oxford-iiit-pet下面的trainval.txt和test.txt切分的 这两个文本文档是官方给的 至于为什么少两个类别 我也不清楚 不过并妨碍大家理解任务和熟悉仓库 昂~

### 启动训练和测试可视化
请参考[Vision](../README.md)