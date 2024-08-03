import glob
import os
from os.path import join as opj
import torch
from torch.utils.data import Dataset
import json
from PIL import Image
from pathlib import Path
from collections import defaultdict
from built.class_augmenter import ClassWiseAugmenter
from prettytable import PrettyTable
import datasets

class ImageDatasets(Dataset):
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

        # for multi-label
        images_path_unique = []
        hashtable = defaultdict(list) # image2class

        for cla in data_class:
            cla_path = os.path.join(src_path, cla)
            images = [os.path.join(src_path, cla, i) for i in os.listdir(cla_path)
                      if os.path.splitext(i)[-1] in support]
            image_class = class_indices[cla]
            images_path.extend(images)
            images_label += [image_class] * len(images)

            # multi-label
            for img_path in images:
                img_basename = os.path.basename(img_path)
                # filter the same image name in different class for multi-label strategy, in case of over-sample
                if img_basename not in hashtable:
                    images_path_unique.append(img_path)
                hashtable[img_basename].append(image_class)

        self.hashtable = hashtable
        self.images_unique = images_path_unique
        self.images = images_path
        self.labels = images_label
        self.transforms = transforms
        self.label_transforms = label_transforms
        self.class_indices = data_class
        self.multi_label = False


    def __getitem__(self, idx):
        img = ImageDatasets.read_image(self.images[idx] if not self.multi_label else self.images_unique[idx])
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
    def set_label_transforms(label, num_classes, label_smooth): # idx -> vector
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
        except OSError: # 若图像损坏 使用opencv读
            import cv2
            img = cv2.imread(path)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        return img

    @staticmethod
    def tell_data_distribution(imgdir: str, logger, nc: int):
        data_distribution, mt, mv = [], 'train', 'val'
        train_total, val_total = 0, 0
        for c in os.listdir(opj(imgdir, mt)):
            if c.startswith('.'): continue
            train_len = len(glob.glob(opj(imgdir, mt, c, '*.jpg'))) + len(glob.glob(opj(imgdir, mt, c, '*.png')))
            val_len = len(glob.glob(opj(imgdir, mv, c, '*.jpg'))) + len(glob.glob(opj(imgdir, mv, c, '*.png')))
            data_distribution.append(
                (c, train_len, val_len)
            )
            train_total += train_len
            val_total += val_len
        data_distribution.append(('total', train_total, val_total))

        pretty_table = PrettyTable(['Class', 'Train Samples', 'Val Samples'])
        for row in data_distribution: pretty_table.add_row(row)
        msg = '\n' + str(pretty_table)
        logger.both(msg) if nc <= 50 else logger.log(msg)
        return data_distribution

class PredictImageDatasets(Dataset):
    def __init__(self, root = None, transforms = None, postfix: tuple = ('jpg', 'png')):
        assert transforms is not None, 'transforms would not be None'
        if root is None: # used for face embedding infer
            self.imgs_path = []
        else:
            self.imgs_path = glob.glob(os.path.join(root, f'*.{postfix[0]}')) + glob.glob(os.path.join(root, f'*.{postfix[1]}'))
            assert len(self.imgs_path) != 0, f'there are no files with postfix as {postfix}'
        self.transforms = transforms

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