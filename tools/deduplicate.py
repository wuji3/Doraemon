import os
import imagehash
from PIL import Image
from tqdm import tqdm

def find_similar_images(userpaths, hashfunc=imagehash.average_hash):
    def is_image(filename):
        f = filename.lower()
        return f.endswith('.png') or f.endswith('.jpg') or \
            f.endswith('.jpeg') or f.endswith('.bmp') or \
            f.endswith('.gif') or '.jpg' in f or f.endswith('.svg')

    image_filenames = []
    for userpath in userpaths:
        image_filenames += [os.path.join(userpath, path) for path in os.listdir(userpath) if is_image(path)]
    images = {}
    for img in tqdm(sorted(image_filenames), total=len(image_filenames)):
        try:
            hash = hashfunc(Image.open(img))
        except Exception as e:
            print('Problem:', e, 'with', img)
            continue
        if hash in images:
            # print(img, '  already exists as', ' '.join(images[hash]))
            os.remove(img)  # delete the duplicate image
            feat = f"/home/duke/data/favie/v2-embedding/features/{os.path.basename(img).replace('.jpg', '.npy')}"
            if os.path.isfile(feat):
                os.remove(feat)
        else:
            images[hash] = images.get(hash, []) + [img]

    return images

if __name__ == '__main__':
    userpaths = ['/home/duke/data/favie/v2-embedding/images']
    find_similar_images(userpaths, imagehash.dhash)