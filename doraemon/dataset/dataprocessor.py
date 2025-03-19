from .basedataset import ImageDatasets
from ..built.class_augmenter import ClassWiseAugmenter
import torch
import os
from torch.utils.data import DataLoader
from typing import Optional

class SmartDataProcessor:
    def __init__(self, data_cfgs: dict, rank, project, training: bool = True):
        self.data_cfgs = data_cfgs # root, nw, imgsz, train, val
        self.rank = rank
        self.project = project
        self.label_transforms = None # used in CenterProcessor.__init__

        if training:
            self.train_dataset = self.create_dataset('train')

    def create_dataset(self, mode: str, training: bool = True, id2label: Optional[dict] = None):
        assert mode in {'train', 'val'}

        cfg = self.data_cfgs.get(mode, -1)
        if isinstance(cfg, dict):
            dataset = ImageDatasets(root_or_dataset=self.data_cfgs['root'], mode=mode,
                                    transforms=ClassWiseAugmenter(cfg['augment'], None, None) if mode == 'val' else \
                               ClassWiseAugmenter(cfg['augment'], cfg['class_aug'], cfg['base_aug']),
                                    project=self.project, rank=self.rank, training = training, id2label=id2label)
        else: 
            dataset = None
        return dataset

    def set_augment(self, mode: str, transforms = None): # sequence -> T.Compose([...])
        if transforms is None:
            transforms = self.val_dataset.transforms.base_transforms
        dataset = getattr(self, f'{mode}_dataset')
        dataset.transforms.base_transforms = transforms

    def auto_aug_weaken(self, epoch: int, milestone: int, sequence: Optional[torch.nn.Module] = None):
        if epoch == milestone:
            # sequence = create_AugTransforms('random_horizonflip to_tensor normalize')
            self.set_augment('train', transforms = sequence)

    @staticmethod
    def set_dataloader(dataset, bs: int = 256, nw: int = 0, pin_memory: bool = True, shuffle: bool = True, sampler = None, collate_fn= None, *args, **kwargs):
        assert not (shuffle and sampler is not None)
        nd = torch.cuda.device_count()
        nw = min([os.cpu_count() // max(nd, 1), nw])
        return DataLoader(dataset=dataset, batch_size=bs, num_workers=nw, pin_memory=pin_memory, sampler=sampler, shuffle=shuffle, collate_fn=collate_fn, *args, **kwargs)