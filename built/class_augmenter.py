from typing import Union, List, Dict, Optional
from utils.augment import BaseClassWiseAugmenter
import torchvision.transforms as T

class ClassWiseAugmenter(BaseClassWiseAugmenter):
    def __init__(self, base_transforms: Dict, class_transforms_mapping: Optional[Dict[str, List[int]]], common: List[int]):
        super().__init__(base_transforms=base_transforms, class_transforms_mapping=class_transforms_mapping)
        # common_transforms 需要定制的部分 即通用增强
        if common is not None:
            if isinstance(common, str): common = list(map(int, common.split()))
            self.common_transforms = T.Compose([t for i, t in enumerate(self.base_transforms.transforms) if i in common])
        else: self.common_transforms = common

    def __call__(self, image, label: Union[List, int], class_indices: List[int]):
        if self.class_transforms is None:
            return super().__call__(image=image, label=label, class_indices=class_indices)

        # softmax
        if isinstance(label, int):
            if class_indices[label] in self.class_transforms:
                return self.class_transforms[class_indices[label]](image)
            else: return self.common_transforms(image)
        # sigmoid
        elif isinstance(label, list): # multi-label
            # multi-label
            if len(label) == 1: # 定制类
                c = label[0]
                if class_indices[c] in self.class_transforms:
                    return self.class_transforms[class_indices[c]](image)

            # 非定制
            return self.common_transforms(image)