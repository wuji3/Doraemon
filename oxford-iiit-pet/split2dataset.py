import os
import shutil
from os.path import join as opj
# -------------------------------------用于划分斯坦福宠物数据集--------------------------------------- #

def splitImg2Category(dataDir="images/", resDir="data/"):
    '''
    归类图像到不同目录中
    '''
    for one_pic in os.listdir(dataDir):
        one_path = dataDir + one_pic
        oneDir = opj(resDir, one_pic.split('_')[0].strip())
        os.makedirs(oneDir, exist_ok=True)
        shutil.copy(one_path, opj(oneDir, one_pic))

if __name__ == '__main__':

    # 归类
    splitImg2Category()

    # 划分数据集
    annos = ['annotations/trainval.txt', 'annotations/test.txt' ]

    for i, anno in enumerate(annos):
        mode = 'train' if i == 0 else 'val'

        with open(anno, 'r') as f:
            for img_name in f.readlines():
                img_name = img_name.split()[0].strip()
                img_catogary = img_name.split('_')[0].strip()

                dstDir = opj('pet', mode, img_catogary)
                os.makedirs(opj('pet', mode, img_catogary), exist_ok=True)

                cur_catogary = opj('pet', img_catogary)
                if not os.path.exists(opj(dstDir, img_name + '.jpg')):
                    shutil.move(opj('pet', img_catogary, img_name + '.jpg'), dstDir)

    # 清空
    for dir_ in os.listdir('pet'):
        if dir_ not in ('train', 'val'):
            shutil.rmtree(opj('pet', dir_))