from typing import Union, List, Dict, Optional
from dataset.transforms import BaseClassWiseAugmenter
import torchvision.transforms as T

class ClassWiseAugmenter(BaseClassWiseAugmenter):
    def __init__(self, base_transforms: Dict, class_transforms_mapping: Optional[Dict[str, List[int]]], common: List[int]):
        super().__init__(base_transforms=base_transforms, class_transforms_mapping=class_transforms_mapping)
        # common_transforms
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
            if len(label) == 1: # Customized specific class
                c = label[0]
                if class_indices[c] in self.class_transforms:
                    return self.class_transforms[class_indices[c]](image)

            # Generally common class
            return self.common_transforms(image)