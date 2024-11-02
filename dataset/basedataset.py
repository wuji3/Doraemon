import glob
import os
from os.path import join as opj
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import json
from PIL import Image
from pathlib import Path
from collections import defaultdict
from built.class_augmenter import ClassWiseAugmenter
from prettytable import PrettyTable
import datasets
import numpy as np
from typing import Optional, Union
import io

class ImageDatasets(Dataset):
    def __init__(self, root_or_dataset, mode='train', transforms=None, label_transforms=None, project=None, rank=None, training=True):
        self.transforms = transforms
        self.label_transforms = label_transforms
        self.multi_label = False
        self.is_local_dataset = True
        self.training = training

        try:
            self._init_from_local(root_or_dataset, mode, project, rank)
        except AssertionError:
            try:
                self._init_from_huggingface(root_or_dataset, mode, project, rank)
                self.is_local_dataset = False
            except Exception as e:
                raise ValueError(f"Failed to load dataset from local path or Hugging Face. Error: {str(e)}")
        
    def _init_from_local(self, root, mode, project, rank):
        assert os.path.isdir(root), f"Dataset root: {root} does not exist."
        src_path = os.path.join(root, mode)

        if self.training:
            data_class = [cla for cla in os.listdir(src_path) if os.path.isdir(os.path.join(src_path, cla))]
            data_class.sort()
            class_indices = dict((k, v) for v, k in enumerate(data_class))
            self._save_class_indices(class_indices, mode, project, rank)
        else:
            class_indices = self._load_class_indices(project)
            data_class: list[str] = list(class_indices.keys())

        support = [".jpg", ".png"]

        images_path = []
        images_label = []
        images_path_unique = []
        hashtable = defaultdict(list)

        for cla in data_class:
            cla_path = os.path.join(src_path, cla)
            images = [os.path.join(src_path, cla, i) for i in os.listdir(cla_path)
                      if os.path.splitext(i)[-1].lower() in support]
            image_class = class_indices[cla]
            images_path.extend(images)
            images_label += [image_class] * len(images)

            for img_path in images:
                img_basename = os.path.basename(img_path)
                if img_basename not in hashtable:
                    images_path_unique.append(img_path)
                hashtable[img_basename].append(image_class)

        self.hashtable = hashtable
        self.images_unique = images_path_unique
        self.images = images_path
        self.labels = images_label
        self.class_indices = data_class

    def _init_from_huggingface(self, dataset_name, split, project, rank):
        if split == "val": split = "validation"

        self.dataset = load_dataset(dataset_name, split=split)
        
        if 'label' in self.dataset.features:
            label_feature = self.dataset.features['label']
            if isinstance(label_feature, datasets.ClassLabel):
                data_class: list[str] = label_feature.names
            else:
                raise ValueError("Dataset 'label' feature is not of type ClassLabel")
        else:
            raise ValueError("Dataset does not contain 'label' feature")

        if self.training:
            data_class.sort()
            class_indices = dict((k, v) for v, k in enumerate(data_class))
            self._save_class_indices(class_indices, split, project, rank)
        else:
            class_indices = self._load_class_indices(project)
            data_class = list(class_indices.keys())

        self.images = self.dataset['image']
        self.labels = self.dataset['label']

        self.hashtable = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.hashtable[idx].append(label)

        self.images_unique = list(range(len(self.images)))
        self.class_indices = data_class

    def _save_class_indices(self, class_indices, mode, project, rank):
        if mode in ("val", "validation"): return
        if rank in {-1, 0}:
            json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
            if project is not None:
                os.makedirs('./run', exist_ok=True)
                with open(Path(project) / 'class_indices.json', 'w') as json_file:
                    json_file.write(json_str)

    def _load_class_indices(self, project):
        class_indices_path = Path(project) / 'class_indices.json'
        if not class_indices_path.exists():
            raise FileNotFoundError(f"Class indices file not found at {class_indices_path}")
        
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
        
        return {v: int(k) for k, v in class_indices.items()}

    def __getitem__(self, idx):
        if hasattr(self, 'dataset'):  # Hugging Face dataset
            img = self.images[idx]
            label = self.labels[idx]
            if isinstance(img, Image.Image):
                if img.mode != 'RGB':
                    img = img.convert('RGB')
            elif isinstance(img, np.ndarray):
                if img.shape[-1] != 3:
                    img = Image.fromarray(img).convert('RGB')
            else:
                raise ValueError(f"Unexpected image type: {type(img)}")
        else:  # Local dataset
            img = self.read_image(self.images[idx] if not self.multi_label else self.images_unique[idx])
            label = self.hashtable[os.path.basename(self.images[idx])] if self.multi_label else self.labels[idx]

        if self.transforms is not None:
            img = self.transforms(img, label, self.class_indices) if type(self.transforms) is ClassWiseAugmenter else self.transforms(img)

        if self.label_transforms is not None:
            label = self.label_transforms(label)

        return img, label

    def __len__(self):
        return len(self.images) if not self.multi_label else len(self.images_unique)

    @staticmethod
    def collate_fn(batch):
        imgs, labels = tuple(zip(*batch))
        imgs = torch.stack(imgs, dim=0)
        labels = torch.as_tensor(labels) if isinstance(labels[0], int) else torch.stack(labels, dim=0)
        return imgs, labels

    @staticmethod
    def set_label_transforms(label, num_classes, label_smooth):
        vector = torch.zeros(num_classes).fill_(0.5 * label_smooth)
        if isinstance(label, int):
            vector[label] = 1 - 0.5 * label_smooth
        elif isinstance(label, list):
            vector = torch.scatter(vector, dim=0, index=torch.as_tensor(label), value=1 - 0.5 * label_smooth)
        return vector

    @staticmethod
    def read_image(path: str):
        try:
            img = Image.open(path).convert('RGB')
        except OSError:
            import cv2
            img = cv2.imread(path)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img
    
    @staticmethod
    def tell_data_distribution(datasets, logger, nc: int, is_local_dataset: bool):
        data_distribution = defaultdict(lambda: {'train': 0, 'val': 0})
        
        for split, dataset in datasets.items():
            for label in dataset.labels:
                class_name = dataset.class_indices[label]
                data_distribution[class_name][split] += 1

        pretty_table = PrettyTable(['Class', 'Train Samples', 'Val Samples'])
        train_total, val_total = 0, 0
        
        for class_name, counts in data_distribution.items():
            train_count = counts['train']
            val_count = counts['val']
            pretty_table.add_row([class_name, train_count, val_count])
            train_total += train_count
            val_total += val_count

        pretty_table.add_row(['total', train_total, val_total])

        msg = '\n' + str(pretty_table)
        logger.both(msg) if nc <= 50 else logger.log(msg)
        return list(data_distribution.items())


class PredictImageDatasets(Dataset):
    def __init__(self, root = None, transforms = None, postfix: tuple = ('jpg', 'png'), sampling: Optional[int] = None):
        assert transforms is not None, 'transforms would not be None'
        if root is None: # used for face embedding infer
            self.imgs_path = []
        else:
            if not os.path.isdir(root): 
                raise ValueError(f"The provided path {root} is not a directory. If you're trying to use a Hugging Face dataset, please note that only supports local datasets now")

            self.imgs_path = glob.glob(os.path.join(root, f'*.{postfix[0]}')) + glob.glob(os.path.join(root, f'*.{postfix[1]}'))
            assert len(self.imgs_path) != 0, f'there are no files with postfix as {postfix}'
        self.transforms = transforms

        if sampling is not None:
            self.imgs_path = self.imgs_path[:sampling]

    def __getitem__(self, idx: int):
        img = ImageDatasets.read_image(self.imgs_path[idx])

        tensor = self.transforms(img)

        return img, tensor, self.imgs_path[idx]

    def __len__(self):
        return len(self.imgs_path)

    @staticmethod
    def collate_fn(batch):
        images, tensors, image_path = tuple(zip(*batch))
        return images, torch.stack(tensors, dim=0), image_path

class CBIRDatasets(Dataset):
    def __init__(self, 
                 root: str, 
                 transforms = None,
                 postfix: tuple = ('jpg', 'png'),
                 mode: str = 'query'):

        assert transforms is not None, 'transforms would not be None'
        assert mode in ('query', 'gallery'), 'make sure mode is query or gallery'
        query_dir, gallery_dir = os.path.join(opj(root, 'query')), os.path.join(opj(root, 'gallery'))
        assert os.path.isdir(query_dir) and os.path.isdir(gallery_dir), 'make sure query dir and gallery dir exists'

        is_subset, query_identity, gallery_identity = self._check_subset(query_dir, gallery_dir) 
        if not is_subset:
            raise ValueError('query identity is not subset of gallery identity')

        data = {'query': [], 'pos': []}
        gallery = {'gallery': []}
        if mode == 'query':
            for q in query_identity:
                one_identity_queries = glob.glob(opj(query_dir, q, f'*.{postfix[0]}')) + glob.glob(opj(query_dir, q, f'*.{postfix[1]}'))
                one_identity_positives = glob.glob(opj(gallery_dir, q, f'*.{postfix[0]}')) + glob.glob(opj(gallery_dir, q, f'*.{postfix[1]}'))
                for one_q in one_identity_queries:
                    data['query'].append(one_q)
                    data['pos'].append(one_identity_positives)
        else:
            gallery['gallery'] = glob.glob(opj(gallery_dir, f'**/*.{postfix[0]}')) + glob.glob(opj(gallery_dir, f'**/*.{postfix[1]}'))
        
        self.mode = mode
        self.data = datasets.Dataset.from_dict(data)
        self.gallery = datasets.Dataset.from_dict(gallery)

        self.transforms = transforms
    
    @classmethod
    def build(cls, root: str, transforms = None, postfix: tuple = ('jpg', 'png')):
        return cls(root, transforms, postfix, 'query'), cls(root, transforms, postfix, 'gallery')

    def _check_subset(self, query: str, gallery: str):
        query_identity = [q for q in os.listdir(query) if not q.startswith('.')]
        gallery_identity = [q for q in os.listdir(gallery) if not q.startswith('.')]

        return set(query_identity).issubset(set(gallery_identity)), query_identity, gallery_identity
    
    def __getitem__(self, idx: int):
        data = self.data[idx]['query'] if self.mode == 'query' else self.gallery[idx]['gallery']
        data_image = ImageDatasets.read_image(data)
        tensor = self.transforms(data_image)

        return tensor     
    
    def __len__(self):
        return self.data.num_rows if self.mode == 'query' else self.gallery.num_rows

class EmbeddingDistillDataset(Dataset):
    def __init__(self, 
                 image_dir: str,
                 feat_dir: str,
                 transform = None,
                 exclude = None) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.feat_dir = feat_dir
        self.transform = transform
        self.images, self.labels = [], []

        if exclude is not None:
            with open(exclude, 'r') as f:
                exclude_files = f.readlines()
                exclude_files = [path.strip() for path in exclude_files]
                exclude_files = set(exclude_files)
        # Collect all valid images and corresponding .npy files
        for img_path in EmbeddingDistillDataset.generator(image_dir, 'jpg'):
            basename = os.path.splitext(os.path.basename(img_path))[0]
            feat_path = os.path.join(feat_dir, f'{basename}.npy')
            
            if os.path.isfile(feat_path):
                if exclude is None:
                    self.images.append(img_path)
                    self.labels.append(feat_path)
                else:
                    if feat_path not in exclude_files:
                        self.images.append(img_path)
                        self.labels.append(feat_path) 

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = ImageDatasets.read_image(img_path)
        
        # Apply transforms to the image if any
        if self.transform is not None:
            image = self.transform(image)
        
        # Load corresponding feature from .npy file
        feat_path = self.labels[idx]
        feature = np.load(feat_path)
        
        return image, feature

    @staticmethod
    def generator(image_dir, post_fix = 'jpg'):
        with os.scandir(image_dir) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(f".{post_fix}"):
                    yield entry.path
    
    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        return torch.stack(images, dim=0), torch.from_numpy(np.stack(labels, axis=0))