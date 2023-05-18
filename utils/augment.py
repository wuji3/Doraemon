import copy
from PIL import Image
from typing import Optional, List, Tuple, Callable
import numpy as np
import random
import torchvision.transforms as T
from functools import wraps
import torch.nn as nn

# all methods based on PIL
__all__ = ['color_jitter', 'random_color_jitter', 'random_horizonflip', 'random_verticalflip', 'to_tensor', 'to_tensor_without_div','normalize',
           'random_augment', 'center_crop', 'resize', 'centercrop_resize', 'random_cutout','random_affine','create_AugTransforms']

_imgsz_related_methods = {'center_crop', 'resize', 'centercrop_resize'}
class _RandomApply: # decorator
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            if random.random() < self.prob:
                return func(*args,**kwargs)
            return lambda x: x
        return wrapper

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
        img_h = img.size[1]
        img_w = img.size[0]
        h = self.h_range if self.h_range is not None else [0, img_h] # PIL Image size->(w,h)
        w = self.w_range if self.w_range is not None else [0, img_w]

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

class CenterCropAndResize(nn.Sequential):
    def __init__(self, center_size, re_size):
        super().__init__(T.CenterCrop(center_size),
                         T.Resize(re_size))

class RandomColorJitter(T.ColorJitter):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def forward(self, img):
        r = random.random()
        if r < 0.2:
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
def color_jitter(brightness: float = 0.1,
                 contrast: float = 0.1,
                 saturation: float = 0.1,
                 hue: float = 0.1):
    return T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

@register_method
def random_color_jitter(brightness: float = 0.1,
                 contrast: float = 0.1,
                 saturation: float = 0.1,
                 hue: float = 0.1):
    return RandomColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

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
    return T.Normalize(mean=mean, std=std)

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
    return T.Resize(size = size)

@register_method
def centercrop_resize(size):
    center_size, re_size = size
    return CenterCropAndResize(center_size, re_size)

@register_method
def random_affine(degrees = 0., translate = 0., scale = 0., shear = 0., fill=0, center=None):
    return T.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear, fill=fill, center=center)

def create_AugTransforms(augments: str, imgsz = None):
    augments = augments.strip().split()
    return T.Compose(tuple(map(lambda x: AUG_METHODS[x]() if x not in _imgsz_related_methods else AUG_METHODS[x](imgsz), augments)))

def list_augments():
    augments = [k for k, v in AUG_METHODS.items()]
    return sorted(augments)