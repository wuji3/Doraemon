import copy
from PIL import Image
from typing import Optional, List, Tuple, Callable
import numpy as np
import random
import torchvision.transforms as T
from functools import wraps
import torch.nn as nn
import glob
import os
from torchvision.transforms.functional import InterpolationMode

# all methods based on PIL
__all__ = ['color_jitter', # 颜色抖动
           'random_color_jitter',# [随机]颜色抖动
           'random_horizonflip', # [随机]水平翻转
           'random_verticalflip', # [随机]上下翻转
           'random_crop', # [随机]抠图
           'random_augment', # RandAug
           'center_crop', # 中心抠图
           'resize', # 缩放
           'centercrop_resize', # 中心抠图+缩放
           'random_cutout', # 随机CutOut
           'random_cutaddnoise', # 随机CutOut+增加噪音
           'random_affine', # 随机仿射变换
           'to_tensor', # 转Tensor
           'to_tensor_without_div', # 转Tensor不除255
           'normalize', # Normalize
           'random_gaussianblur', # 随机高斯模糊
           'create_AugTransforms',
           'list_augments']

AUG_METHODS = {}
def register_method(fn: Callable):
    key = fn.__name__
    if key in AUG_METHODS:
        raise ValueError(f"An entry is already registered under the name '{key}'.")
    AUG_METHODS[key] = fn
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper

class Cutout:
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes: int, length: int, ratio: float,
                 h_range: Optional[List[int]] = None, w_range: Optional[List[int]] = None,
                 prob: float = 0.5):
        self.n_holes = n_holes
        self.length = length
        self.ratio = ratio
        self.h_range = h_range
        self.w_range = w_range
        self.prob = prob

    def __call__(self, image):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W) from PIL
        Returns:
            PIL: Image with n_holes of dimension length x length cut out of it.
        """
        if random.random() > self.prob:
            return image
        img = copy.deepcopy(image) # protect source image

        h = self.h_range if self.h_range is not None else [0, img.height] # PIL Image size->(w,h)
        w = self.w_range if self.w_range is not None else [0, img.width]

        mask_w = int(random.uniform(1-self.ratio, 1+self.ratio) * self.length)
        mask_h = self.length
        mask = Image.new('RGB', size=(mask_w, mask_h), color=0)

        for n in range(self.n_holes):
            # center
            y = np.random.randint(*h)
            x = np.random.randint(*w)

            # left-up
            x1 = max(0, x - self.length // 2)
            y1 = max(0, y - self.length // 2)

            img.paste(mask, (x1, y1))

        return  img

class CutAddNoise:
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes: int, length: int, noisy_src: str,
                 h_range: Optional[List[int]] = None, w_range: Optional[List[int]] = None,
                 prob: float = 0.5, ):
        self.n_holes = n_holes
        self.length = length
        self.h_range = h_range
        self.w_range = w_range
        self.prob = prob
        self.noisy = glob.glob(f'{noisy_src}/*.jpg')
        assert os.path.splitext(self.noisy[0])[-1] == '.jpg', 'only support .jpg'

    def __call__(self, image):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W) from PIL
        Returns:
            PIL: Image with n_holes of dimension length x length cut out of it.
        """
        if random.random() > self.prob:
            return image
        img = copy.deepcopy(image)  # protect source image

        h = self.h_range if self.h_range is not None else [0, img.height]  # PIL Image size->(w,h)
        w = self.w_range if self.w_range is not None else [0, img.width]

        noisy_image = Image.open(random.choice(self.noisy)).convert('RGB')
        noisy_image = noisy_image.resize(size=(image.width, image.height))

        for n in range(self.n_holes):
            # center
            y = np.random.randint(*h)
            x = np.random.randint(*w)

            # left-up
            x1 = max(0, x - self.length // 2)
            y1 = max(0, y - self.length // 2)

            # right-bottom
            x2 = min(noisy_image.width, x + self.length // 2)
            y2 = min(noisy_image.height, y + self.length // 2)

            noisy_box = noisy_image.crop((x1, y1, x2, y2))
            img.paste(noisy_box, (x1, y1))

        return img

class CenterCropAndResize(nn.Sequential):
    def __init__(self, center_size, re_size):
        super().__init__(T.CenterCrop(center_size),
                         T.Resize(re_size, interpolation=InterpolationMode.BILINEAR))

class RandomColorJitter(T.ColorJitter):
    def __init__(self, prob: float = 0.5, *args, **kargs):
        super().__init__(*args, **kargs)
        self.prob = prob

    def forward(self, img):
        r = random.random()
        if r < self.prob:
            return super().forward(img)
        else: return img

class PILToTensorNoDiv:
    def __init__(self):
        self.pil2tensor = T.PILToTensor()

    def __call__(self, pic):
        return self.pil2tensor(pic).float()

@register_method
def random_cutout(n_holes:int = 1, length: int = 200, ratio: float = 0.2,
                  h_range: Optional[List[int]] = None, w_range: Optional[List[int]] = None, prob: float = 0.5):
    return Cutout(n_holes, length, ratio, h_range, w_range, prob)

@register_method
def random_cutaddnoise(n_holes:int = 1, length: int = 200, noisy_src: str = None,
                  h_range: Optional[List[int]] = None, w_range: Optional[List[int]] = None, prob: float = 0.5):
    return CutAddNoise(n_holes, length, noisy_src, h_range, w_range, prob)

@register_method
def color_jitter(brightness: float = 0.1,
                 contrast: float = 0.1,
                 saturation: float = 0.1,
                 hue: float = 0.1):
    return T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
@register_method
def random_crop(*args, **kwargs):
    return T.RandomCrop(*args, **kwargs)
@register_method
def random_color_jitter(prob: float = 0.5, *args, **kwargs):
    # brightness: float = 0.1, contrast: float = 0.1, saturation: float = 0.1, hue: float = 0.1
    return RandomColorJitter(prob = prob, *args, **kwargs)

@register_method
def random_horizonflip(p: float = 0.5):
    return T.RandomHorizontalFlip(p=p)

@register_method
def random_verticalflip(p: float = 0.5):
    return T.RandomVerticalFlip(p=p)

@register_method
def to_tensor():
    return T.ToTensor()

@register_method
def to_tensor_without_div():
    return PILToTensorNoDiv()

@register_method
def normalize(mean: Tuple = (0.485, 0.456, 0.406), std: Tuple = (0.229, 0.224, 0.225)):
    return T.Normalize(mean=mean if isinstance(mean, tuple) else eval(mean),
                       std=std if isinstance(std, tuple) else eval(std))

@register_method
def random_augment(num_ops: int = 2, magnitude: int = 9, num_magnitude_bins: int = 31,):
    return T.RandAugment(num_ops=num_ops, magnitude=magnitude, num_magnitude_bins=num_magnitude_bins)

@register_method
def center_crop(size):
    # size (sequence or int): Desired output size of the crop. If size is an
    # int instead of sequence like (h, w), a square crop (size, size) is
    # made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    return T.CenterCrop(size=size)

@register_method
def resize(size = 224):
    # size (sequence or int) -> square or rectangle: Desired output size. If size is a sequence like
    # (h, w), output size will be matched to this. If size is an int,smaller
    # edge of the image will be matched to this number. i.e,
    # if height > width, then image will be rescaled to (size * height / width, size).
    return T.Resize(size = size, interpolation=InterpolationMode.NEAREST)

@register_method
def centercrop_resize(center_size: tuple, re_size: tuple):
    return CenterCropAndResize(center_size, re_size)

@register_method
def random_affine(degrees = 0., translate = 0., scale = 0., shear = 0., fill=0, center=None):
    return T.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear, fill=fill, center=center)

@register_method
def random_gaussianblur(prob: float = 0.5, kernel_size=3, sigma=(0.1, 2.0)): # 每次transform sigma会均匀采样一次 除非传sigma是固定值
    return T.RandomApply([T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)], p = prob)

def create_AugTransforms(augments: dict):
    augs = []
    for key, params in augments.items():
        if params == 'no_params':
            augs.append(AUG_METHODS[key]())
        else: augs.append(AUG_METHODS[key](**params))
    return T.Compose(augs)
    # augments = augments.strip().split()
    # return T.Compose(tuple(map(lambda x: AUG_METHODS[x](**kwargs) if x not in _imgsz_related_methods else AUG_METHODS[x](imgsz, **kwargs), augments)))

def list_augments():
    augments = [k for k, v in AUG_METHODS.items()]
    return sorted(augments)