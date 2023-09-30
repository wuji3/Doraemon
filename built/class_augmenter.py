from typing import Union, List, Dict, Optional
from utils.augment import BaseClassWiseAugmenter
from PIL.Image import Image
import torchvision.transforms as T

class ClassWiseAugmenter(BaseClassWiseAugmenter):
    def __init__(self, base_transforms: Dict, class_transforms_mapping: Optional[Dict[str, List[int]]], common: List[int]):
        super().__init__(base_transforms=base_transforms, class_transforms_mapping=class_transforms_mapping)
        # common_transforms 需要定制的部分 即通用增强
        if common is not None:
            if isinstance(common, str): common = list(map(int, common.split()))
            self.common_transforms = T.Compose([t for i, t in enumerate(self.base_transforms.transforms) if i in common])
        else: self.common_transforms = common

    def __call__(self, image: Image, label: Union[List, int], class_indices: List[int]):
        if self.class_transforms is None:
            return super().__call__(image=image, label=label, class_indices=class_indices)

        # 下面是需要定制的部分
        if isinstance(label, int):
            return self.class_transforms[class_indices[label]](image)
        elif isinstance(label, list): # multi-label
            # multi-label
            if sum(label) == 1: # 严格的S、B+和B-
                c = label.index(1)
                return self.class_transforms[class_indices[c]](image)
            else: # B+和模糊边界的样本
                return self.common_transforms(image)