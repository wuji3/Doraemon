from dataset.transforms import create_AugTransforms
from PIL import Image
import numpy as np

augs = {
    'random_color_jitter': dict(brightness=0.1, contrast = 0.1, saturation = 0.1, hue = 0.1),
    # 'random_equalize': 'no_params',
    'random_crop_and_resize': dict(size = 640, scale = (0.2, 1.0)),
    'random_cutout': dict(n_holes=4, length=50, prob=0.1, ),
    'random_grayscale': 'no_params',
    # 'random_gaussianblur': dict(kernel_size=5),
    'random_localgaussian': dict(ksize = (37, 37))
}

t = create_AugTransforms(augs)

img = Image.open('/Users/duke/vision-classifier/data/train/S/gds_level40_gds_id21271181_spu_id367361.png')

images = []
for i in range(1, 29):
    filename = f'image_{i}.jpg'
    image = t(img)
    images.append(image)

# 将图像列表转换为 numpy 数组
array_images = [np.array(image) for image in images]

# 创建一个 rows行 cols列的图像网格
rows, cols = 4, 7
grid = np.zeros((rows * array_images[0].shape[0], cols * array_images[0].shape[1], 3), dtype=np.uint8)

# 在图像网格中排列图像
for r in range(rows):
    for c in range(cols):
        image_index = r * cols + c
        if image_index < len(array_images):
            grid[r*array_images[0].shape[0]:(r+1)*array_images[0].shape[0], c*array_images[0].shape[1]:(c+1)*array_images[0].shape[1], :] = array_images[image_index]

# 将图像网格转换为 PIL.Image 对象并显示
Image.fromarray(grid).show()