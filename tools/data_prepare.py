import glob
import argparse
import os
from os.path import join as opj
import shutil
import pandas as pd
from typing import Union

def parse_opt():
    parsers = argparse.ArgumentParser()
    parsers.add_argument('--postfix', default='jpg', help='图像文件后缀')
    parsers.add_argument('--root', default='data', help='数据路径')
    parsers.add_argument('--frac', type=float, nargs='+', help='训练集占比')
    parsers.add_argument('--drop', action='store_true')

    return parsers.parse_args()

def data_split(postfix: str, root: str, frac: Union[float, list], drop: bool):

    # 过滤隐藏文件
    all_classes = [x for x in os.listdir(root) if not x.startswith('.')]
    all_classes.sort()

    if len(frac) > 1: assert len(frac) == len(all_classes), '若传多个frac 需保证每个类都要有一个frac len(frac) == len(all_classes)'
    else:
        a = frac[0]
        frac = [a for _ in all_classes]

    modes = ['train', 'val']
    for m in modes:
        os.makedirs(opj(root, m), exist_ok=True)

    for i, cls in enumerate(all_classes):
        for m in modes:
            os.makedirs(opj(root, m, cls), exist_ok=True)

        s = pd.Series(glob.glob(opj(root, cls, f'*.{postfix}')))
        train = s.sample(frac=frac[i])
        val = s[~s.isin(train)]

        train.apply(lambda x: shutil.copy(x, opj(root, 'train', cls)))
        val.apply(lambda x: shutil.copy(x, opj(root, 'val', cls)))

        if drop:
            shutil.rmtree(opj(root, cls))

        print(opj(root, cls), '  完成')

if __name__ == '__main__':
    opt = parse_opt()
    data_split(opt.postfix, opt.root, opt.frac, opt.drop)