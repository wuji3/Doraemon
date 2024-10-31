import glob
import shutil
import argparse
import os
from os.path import join as opj

"""
Before
--data
    --ID1
        --xxx1.jpg
        --xxx2.jpg
    --ID2
        --xxx3.jpg
        --xxx4.jpg

After
--data
    --data-query
        --ID1
            --xxx1.jpg
        --ID2
            --xxx3.jpg
    --data-gallery
        --ID1
            --xxx2.jpg
        --ID2
            --xxx4.jpg
"""

def parse_opt():
    parsers = argparse.ArgumentParser()
    parsers.add_argument('--src', default='data', help='Image dir')
    parsers.add_argument('--frac', type=float, help='Fraction of query/gallery')
    parsers.add_argument('--drop', action='store_true', help="Cleaning up the source directory")

    return parsers.parse_args()


def main(opt):
    src = opt.src
    frac = opt.frac
    drop = opt.drop

    src = os.path.realpath(src)
    root = os.path.dirname(src)
    basename = os.path.basename(src)

    all_classes = [x for x in os.listdir(src) if not x.startswith('.')]
    all_classes.sort()

    for c in all_classes:
        os.makedirs(opj(root, f'{basename}-query', c), exist_ok=True)
        os.makedirs(opj(root, f'{basename}-gallery', c), exist_ok=True)

        all_files = glob.glob(opj(src, c, '*'))
        all_files.sort()

        n = len(all_files)
        if n == 1: continue
        else:
            n_query = int(n * frac) if int(n * frac) != 0 else 1

        query_files = all_files[:n_query]
        gallery_files = all_files[n_query:]

        for f in query_files:
            shutil.copy(f, opj(root, f'{src}-query', c, os.path.basename(f)))

        for f in gallery_files:
            shutil.copy(f, opj(root, f'{src}-gallery', c, os.path.basename(f)))

    if drop:
        shutil.rmtree(src)

if __name__ == "__main__":
    main(parse_opt())