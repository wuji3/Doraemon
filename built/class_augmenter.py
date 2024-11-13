from typing import Union, List, Dict, Optional
from dataset.transforms import BaseClassWiseAugmenter

class ClassWiseAugmenter(BaseClassWiseAugmenter):
    def __init__(self, base_transforms: Dict, class_transforms_mapping: Optional[Dict[str, List[int]]], base: List[int]):
        if base is not None:
            assert isinstance(base, list), f'{base} is not a list of indices'
            base_transforms = [t for i, t in enumerate(base_transforms) if i in base]

        super().__init__(base_transforms=base_transforms, class_transforms_mapping=class_transforms_mapping)

    def __call__(self, image, label: Union[List, int], class_indices: List[int]):
        if self.class_transforms is None:
            return super().__call__(image=image, label=label, class_indices=class_indices)

        # softmax
        if isinstance(label, int):
            if class_indices[label] in self.class_transforms:
                return self.class_transforms[class_indices[label]](image)
            else: return self.base_transforms(image)
        # sigmoid
        elif isinstance(label, list): # multi-label
            # multi-label
            if len(label) == 1: # Customized specific class
                c = label[0]
                if class_indices[c] in self.class_transforms:
                    return self.class_transforms[class_indices[c]](image)

            # Generally common class
            return self.base_transforms(image)