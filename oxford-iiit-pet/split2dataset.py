import os
import shutil
from os.path import join as opj

# -------------------------------------tag class--------------------------------------- #
def splitImg2Category(dataDir="oxford-iiit-pet/images/", resDir="pet/"):
    '''
    Give tag to images
    '''
    for one_pic in os.listdir(dataDir):
        one_path = dataDir + one_pic
        oneDir = opj(resDir, one_pic.split('_')[0].strip())
        os.makedirs(oneDir, exist_ok=True)
        shutil.copy(one_path, opj(oneDir, one_pic))

if __name__ == '__main__':

    # tag class
    splitImg2Category()

    # split
    annos = ['oxford-iiit-pet/annotations/trainval.txt', 'oxford-iiit-pet/annotations/test.txt' ]

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

    # remove
    for dir_ in os.listdir('pet'):
        if dir_ not in ('train', 'val'):
            shutil.rmtree(opj('pet', dir_))