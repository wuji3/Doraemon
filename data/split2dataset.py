import os
import shutil
from os.path import join as opj

def splitImg2Category(dataDir="oxford-iiit-pet/images/", resDir="pet/"):
    for one_pic in os.listdir(dataDir):
        one_path = opj(dataDir, one_pic)
        oneDir = opj(resDir, one_pic.split('_')[0].strip())
        os.makedirs(oneDir, exist_ok=True)
        shutil.copy(one_path, opj(oneDir, one_pic))

if __name__ == '__main__':
    # tag class
    splitImg2Category()

    # split
    annos = ['oxford-iiit-pet/annotations/trainval.txt', 'oxford-iiit-pet/annotations/test.txt']

    for i, anno in enumerate(annos):
        mode = 'train' if i == 0 else 'val'

        with open(anno, 'r') as f:
            for img_name in f.readlines():
                img_name = img_name.split()[0].strip()
                img_category = img_name.split('_')[0].strip()

                dstDir = opj('pet', mode, img_category)
                os.makedirs(dstDir, exist_ok=True)

                cur_category = opj('pet', img_category)
                if not os.path.exists(opj(dstDir, img_name + '.jpg')):
                    shutil.move(opj('pet', img_category, img_name + '.jpg'), dstDir)

    # remove temporary directories
    for dir_ in os.listdir('pet'):
        if dir_ not in ('train', 'val'):
            shutil.rmtree(opj('pet', dir_))

    # remove original oxford-iiit-pet directory
    shutil.rmtree('oxford-iiit-pet')

    print("Data preparation completed. The 'pet' directory is ready for use.")
