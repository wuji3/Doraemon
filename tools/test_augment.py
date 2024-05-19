from dataset.transforms import create_AugTransforms
from PIL import Image
import numpy as np
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--img_path', type=str, default='/Users/wuji/Desktop/pujing.jpg', help='Path of raw image')
    parser.add_argument('-o', '--output_path', type=str, default='save_img.jpg', help='Path to save image')
    parser.add_argument('-H', '--height', default=4, help='Height of the jagged grid')
    parser.add_argument('-W', '--width', default=7, help='Width of the jagged grid')

    return parser.parse_args()

def create_augs():
    augs = {
        'random_color_jitter': dict(brightness=0.1, contrast = 0.1, saturation = 0.1, hue = 0.1),
        # 'random_equalize': 'no_params',
        'random_crop_and_resize': dict(size = 520, scale = (0.2, 1.0)),
        'random_cutout': dict(n_holes=4, length=50, prob=0.1, ),
        'random_grayscale': 'no_params',
        # 'random_gaussianblur': dict(kernel_size=5),
        'random_localgaussian': dict(ksize = (37, 37))
    }

    return augs

def main(args):

    augs = create_augs()

    t = create_AugTransforms(augs)

    img = Image.open(args.img_path).convert("RGB")

    images = []
    for i in range(1, args.height * args.width + 1):
        image = t(img)
        images.append(image)

    array_images = [np.array(image) for image in images]

    rows, cols = args.height, args.width
    grid = np.zeros((rows * array_images[0].shape[0], cols * array_images[0].shape[1], 3), dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            image_index = r * cols + c
            if image_index < len(array_images):
                grid[r*array_images[0].shape[0]:(r+1)*array_images[0].shape[0], c*array_images[0].shape[1]:(c+1)*array_images[0].shape[1], :] = array_images[image_index]

    img_pillow = Image.fromarray(grid)
    img_pillow.show()
    img_pillow.save(args.output_path)
    
if __name__ == '__main__':
    main(parse_opt())