import glob
import os
import torch
from torch.utils.data import Dataset
import json
from PIL import Image
from pathlib import Path
from collections import defaultdict
from built.class_augmenter import ClassWiseAugmenter

class BaseDatasets(Dataset):
    def __init__(self, root, mode, transforms = None, label_transforms = None, project = None, rank = None):
        assert os.path.isdir(root), "dataset root: {} does not exist.".format(root)
        src_path = os.path.join(root, mode)
        data_class = [cla for cla in os.listdir(src_path) if os.path.isdir(os.path.join(src_path, cla))]
        # sort
        data_class.sort()

        class_indices = dict((k, v) for v, k in enumerate(data_class))
        if rank in {-1, 0}:
            json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
            os.makedirs('./run', exist_ok=True)
            if project is not None:
                with open(Path(project) / 'class_indices.json', 'w') as json_file:
                    json_file.write(json_str)

        support = [".jpg", ".png"]

        images_path = []  # image path
        images_label = []  # label idx

        hashtable = defaultdict(list)
        for cla in data_class:
            cla_path = os.path.join(src_path, cla)
            images = [os.path.join(src_path, cla, i) for i in os.listdir(cla_path)
                      if os.path.splitext(i)[-1] in support]
            image_class = class_indices[cla]
            for img_path in images:
                img_basename = os.path.basename(img_path)
                # 若multi_label 必有重复样本 这里过滤 若不是multi_label 不会有重复样本 不会过滤掉有效样本
                if img_basename not in hashtable:
                    images_path.append(img_path)
                    images_label.append(image_class)
                hashtable[img_basename].append(image_class)

        self.hashtable = hashtable
        self.images = images_path
        self.labels = images_label
        self.transforms = transforms
        self.label_transforms = label_transforms
        self.class_indices = data_class
        self.multi_label = False


    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        label = self.hashtable[os.path.basename(self.images[idx])] if self.multi_label else self.labels[idx]
        if self.transforms is not None:
            if type(self.transforms) is ClassWiseAugmenter: img = self.transforms(img, label, self.class_indices)
            else: img = self.transforms(img)
        if self.label_transforms is not None:
            label = self.label_transforms(label)

        return img, label


    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        imgs, labels = tuple(zip(*batch))

        imgs = torch.stack(imgs, dim=0)
        labels = torch.as_tensor(labels) if isinstance(labels[0], int) else torch.stack(labels, dim=0)
        return imgs, labels

    @staticmethod
    def set_label_transforms(label, num_classes, label_smooth): # idx -> vector
        vector = torch.zeros(num_classes).fill_(0.5 * label_smooth)
        if isinstance(label, int):
            vector[label] = 1 - 0.5 * label_smooth
        elif isinstance(label, list):
            vector = torch.scatter(vector, dim=0, index=torch.as_tensor(label), value=1 - 0.5 * label_smooth)

        return vector

class PredictDatasets(Dataset):
    def __init__(self, root, transforms = None, postfix: str = 'jpg'):
        assert transforms is not None, 'transforms不能为None'
        self.imgs_path = glob.glob(os.path.join(root, f'*.{postfix}'))
        assert len(self.imgs_path) != 0, f'推理路径内没有{postfix}结尾的文件'
        self.transforms = transforms

    def __getitem__(self, idx: int):
        img = Image.open(self.imgs_path[idx]).convert('RGB')

        tensor = self.transforms(img)

        return img, tensor, self.imgs_path[idx]

    def __len__(self):
        return len(self.imgs_path)
    @staticmethod
    def collate_fn(batch):
        images, tensors, image_path = tuple(zip(*batch))
        return images, torch.stack(tensors, dim=0), image_path