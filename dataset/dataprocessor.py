from dataset.basedataset import ImageDatasets
from dataset.transforms import create_AugTransforms
from built.class_augmenter import ClassWiseAugmenter
import torch
import os
from torch.utils.data import DataLoader

class SmartDataProcessor:
    def __init__(self, data_cfgs: dict, rank, project):
        self.data_cfgs = data_cfgs # root, nw, imgsz, train, val
        self.rank = rank
        self.project = project
        self.label_transforms = None # used in CenterProcessor.__init__

        self.train_dataset = self.create_dataset('train')
        self.val_dataset = self.create_dataset('val')

    def create_dataset(self, mode: str):
        assert mode in {'train', 'val'}

        cfg = self.data_cfgs.get(mode, -1)
        if isinstance(cfg, dict):
            dataset = ImageDatasets(root=self.data_cfgs['root'], mode=mode,
                                    transforms=create_AugTransforms(augments=cfg['augment']) if mode == 'val' else \
                               ClassWiseAugmenter(cfg['augment'], cfg['class_aug'], cfg['common_aug']),
                                    project=self.project, rank=self.rank)
        else: dataset = None
        return dataset

    def set_augment(self, mode: str, sequence = None): # sequence -> T.Compose([...])
        if sequence is None:
            sequence = self.val_dataset.transforms
        dataset = getattr(self, f'{mode}_dataset')
        dataset.transforms = sequence

    def auto_aug_weaken(self, epoch: int, milestone: int):
        if epoch == milestone:
            # sequence = create_AugTransforms('random_horizonflip to_tensor normalize')
            self.set_augment('train', sequence = None)

    @staticmethod
    def set_dataloader(dataset, bs: int = 256, nw: int = 0, pin_memory: bool = True, shuffle: bool = True, sampler = None, collate_fn= None):
        assert not (shuffle and sampler is not None)
        nd = torch.cuda.device_count()
        nw = min([os.cpu_count() // max(nd, 1), nw])
        return DataLoader(dataset=dataset, batch_size=bs, num_workers=nw, pin_memory=pin_memory, sampler=sampler, shuffle=shuffle, collate_fn=collate_fn)