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
from typing import Optional
import pandas as pd

class ImageDatasets(Dataset):
    def __init__(self, root_or_dataset, mode='train', transforms=None, label_transforms=None, project=None, rank=None, training=True):
        self.transforms = transforms
        self.label_transforms = label_transforms
        self.is_local_dataset = True
        self.training = training

        try:
            if os.path.isfile(root_or_dataset) and root_or_dataset.endswith('.csv'):
                self.multi_label = True
                self._init_from_csv(root_or_dataset, mode, project, rank)
            else:
                self.multi_label = False
                self._init_from_local(root_or_dataset, mode, project, rank)
        except AssertionError:
            try:
                self._init_from_huggingface(root_or_dataset, mode, project, rank)
                self.is_local_dataset = False
            except Exception as e:
                raise ValueError(f"Failed to load dataset from local path or Hugging Face. Error: {str(e)}")
        
    def _init_from_csv(self, csv_path, mode, project, rank):
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Filter data based on mode
        is_train = True if mode == 'train' else False
        df = df[df['train'] == is_train].reset_index(drop=True)
        
        # Get image paths
        self.images = df['image_path'].tolist()
        
        # Get class columns (excluding image_path and train columns)
        data_class = [col for col in df.columns if col not in ['image_path', 'train']]
        data_class.sort()
        
        if self.training:
            class_indices = dict((k, v) for v, k in enumerate(data_class))
            self._save_class_indices(class_indices, mode, project, rank)
        else:
            class_indices = self._load_class_indices(project)
            data_class = list(class_indices.keys())
        
        # Vectorize labels
        self.labels = df[data_class].values.tolist()  # shape: (num_samples, num_classes)
        self.class_indices = data_class
    
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

        for cla in data_class:
            cla_path = os.path.join(src_path, cla)
            images = [os.path.join(src_path, cla, i) for i in os.listdir(cla_path)
                      if os.path.splitext(i)[-1].lower() in support]
            image_class = class_indices[cla]
            images_path.extend(images)
            images_label += [image_class] * len(images)

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
            img = ImageDatasets.load_image_from_hf(img)
        else:  # Local dataset
            try:
                img = self.read_image(self.images[idx])
            except Exception as e:
                random_idx = np.random.randint(0, len(self.images))
                while random_idx == idx:
                    random_idx = np.random.randint(0, len(self.images))
                return self.__getitem__(random_idx)
            label = self.labels[idx]

        if self.transforms is not None:
            img = self.transforms(img, label, self.class_indices) if type(self.transforms) is ClassWiseAugmenter else self.transforms(img)

        if self.label_transforms is not None:
            label = self.label_transforms(label)

        return img, label

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        """
        Collate function to handle different types of labels in a batch.
        Supports label smoothing through dataset.label_transforms.
        
        Args:
            batch: List of tuples (image, label)
        
        Returns:
            tuple: (stacked_images, stacked_labels)
        """
        imgs, labels = tuple(zip(*batch))
        imgs = torch.stack(imgs, dim=0)
        
        # Handle different label formats
        if isinstance(labels[0], int):
            # Single-label case: [1,3,0,2,...]
            labels = torch.as_tensor(labels, dtype=torch.int64)
        elif isinstance(labels[0], (list, tuple)):
            # Multi-label case with indices [[0,2], [1,4], [0,1,3], ...]
            labels = torch.stack([torch.as_tensor(lbl, dtype=torch.float) for lbl in labels], dim=0)
        elif isinstance(labels[0], torch.Tensor):
            # Multi-label case (CSV case)
            # Labels should already be processed by set_label_transforms [[1,0,1,0,0], [0,1,0,0,1], ...]
            labels = torch.stack(labels, dim=0).float()
        else:
            raise ValueError(f"Unsupported label type: {type(labels[0])}")
        
        return imgs, labels

    @staticmethod
    def set_label_transforms(label, num_classes, label_smooth):
        """
        Transform labels with label smoothing for both single-label and multi-label cases.
        
        Args:
            label: Label in various formats:
                - int: single-label classification
                - list: multi-label classification (indices)
                - torch.Tensor: multi-label classification (one-hot encoded)
            num_classes: Number of classes
            label_smooth: Label smoothing factor
        
        Returns:
            torch.Tensor: Smoothed label vector
        """
        if isinstance(label, torch.Tensor) and label.size(0) == num_classes:
            # Already one-hot encoded (from CSV)
            if label_smooth > 0:
                # Apply label smoothing: y = y * (1 - α) + α/2
                return label * (1 - label_smooth) + (label_smooth * 0.5)
            return label
        
        # Create smoothed background
        vector = torch.zeros(num_classes).fill_(0.5 * label_smooth)
        
        if isinstance(label, int):
            # Single-label case
            vector[label] = 1 - 0.5 * label_smooth
        elif isinstance(label, (list, tuple)):
            # Multi-label case with indices
            label_tensor = torch.tensor(label)
            indices = torch.nonzero(label_tensor).squeeze()
            vector[indices] = 1 - 0.5 * label_smooth
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
    def load_image_from_hf(image):
        if isinstance(image, Image.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
        elif isinstance(image, np.ndarray):
            if image.shape[-1] != 3:
                image = Image.fromarray(image).convert('RGB')
        else:
            raise ValueError(f"Unexpected image type: {type(image)}") 
        
        return image
    
    @staticmethod
    def tell_data_distribution(datasets, logger, nc: int, is_local_dataset: bool):
        """
        Display the distribution of samples across classes for both train and val sets.
        
        Args:
            datasets: Dictionary containing train and val datasets
            logger: Logger instance for output
            nc: Number of classes
            is_local_dataset: Boolean indicating if dataset is local
        """
        data_distribution = defaultdict(lambda: {'train': 0, 'val': 0})
        
        for split, dataset in datasets.items():
            if hasattr(dataset, 'multi_label') and dataset.multi_label:
                # For multi-label data from CSV
                for label_vector in dataset.labels:
                    # label_vector is a list of 0/1 indicating presence of each class
                    for idx, is_present in enumerate(label_vector):
                        if is_present == 1:
                            class_name = dataset.class_indices[idx]
                            data_distribution[class_name][split] += 1
            else:
                # For single-label data
                for label in dataset.labels:
                    class_name = dataset.class_indices[label]
                    data_distribution[class_name][split] += 1

        # Create and populate the distribution table
        pretty_table = PrettyTable(['Class', 'Train Samples', 'Val Samples'])
        train_total, val_total = 0, 0
        
        # Sort class names to match _init_from_csv order
        sorted_classes = sorted(data_distribution.keys())
        
        # Add rows for each class in sorted order
        for class_name in sorted_classes:
            counts = data_distribution[class_name]
            train_count = counts['train']
            val_count = counts['val']
            pretty_table.add_row([class_name, train_count, val_count])
            train_total += train_count
            val_total += val_count

        # Add total row
        pretty_table.add_row(['total', train_total, val_total])

        # Output the table
        msg = '\n' + str(pretty_table)
        logger.both(msg) if nc <= 50 else logger.log(msg)
        return [(class_name, data_distribution[class_name]) for class_name in sorted_classes]


class PredictImageDatasets(Dataset):
    def __init__(self, root=None, transforms=None, postfix: tuple=('jpg', 'png'), 
                 sampling: Optional[int]=None, class_indices: Optional[list]=None,
                 target_class: Optional[str]=None):
        """
        Dataset for prediction, supporting directory, CSV file, and HuggingFace dataset inputs
        
        Args:
            root: Path to image directory, CSV file, or HuggingFace dataset name
            transforms: Image transformations
            postfix: Tuple of image extensions (for directory mode)
            sampling: Number of samples to use (for subset testing)
            class_indices: List of class names
            target_class: Filter dataset to only include specific class
        """
        assert transforms is not None, 'transforms would not be None'
        self.transforms = transforms
        self.class_indices = class_indices
        self.is_local_dataset = True
        self.multi_label = False
        self.target_class = target_class

        if root is None:  # used for face embedding infer
            self.images = []
        else:
            try:
                if os.path.isfile(root) and root.endswith('.csv'):
                    self.multi_label = True
                    self._init_from_csv(root)
                else:
                    self._init_from_dir(root, postfix)
            except (ValueError, FileNotFoundError):
                try:
                    self._init_from_huggingface(root)
                    self.is_local_dataset = False
                except Exception as e:
                    raise ValueError(f"Failed to load dataset from {root}. Error: {str(e)}")

        if sampling is not None:
            self.images = self.images[:sampling]

    def _init_from_csv(self, csv_path):
        """Initialize dataset from CSV file"""
        df = pd.read_csv(csv_path)
        assert 'image_path' in df.columns, 'CSV must contain image_path column'
        
        # Filter by target_class if specified
        if self.target_class is not None:
            assert self.target_class in df.columns, f'Target class {self.target_class} not found in CSV columns'
            df = df[df[self.target_class] == 1].reset_index(drop=True)
            
        self.images = df['image_path'].tolist()
        assert len(self.images) > 0, 'No valid image paths found in CSV'
        
        # Verify all image paths exist
        for img_path in self.images:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")

    def _init_from_dir(self, root, postfix):
        """Initialize dataset from directory"""
        if not os.path.isdir(root):
            raise ValueError(f"The provided path {root} is not a directory")

        self.images = []
        if self.target_class is not None:
            # If target_class is specified, only look in that directory
            target_dir = os.path.join(root, self.target_class)
            if not os.path.isdir(target_dir):
                raise ValueError(f"Target class directory not found: {target_dir}")
            
            # Search in target directory
            for ext in postfix:
                self.images.extend(glob.glob(os.path.join(target_dir, f'*.{ext}')))
        else:
            # Search recursively in all subdirectories
            for ext in postfix:
                self.images.extend(glob.glob(os.path.join(root, f'**/*.{ext}'), recursive=True))
        
        assert len(self.images) > 0, f'No files found with postfix {postfix}'

    def _init_from_huggingface(self, dataset_name):
        """Initialize dataset from HuggingFace"""
        try:
            from datasets import load_dataset
            # Load validation split by default for prediction
            self.dataset = load_dataset(dataset_name, split='validation')
            
            # Filter by target_class if specified
            if self.target_class is not None:
                if 'label' not in self.dataset.features:
                    raise ValueError("Dataset does not contain 'label' feature")
                
                label_feature = self.dataset.features['label']
                if isinstance(label_feature, datasets.ClassLabel):
                    if self.target_class not in label_feature.names:
                        raise ValueError(f"Target class {self.target_class} not found in dataset classes")
                    target_idx = label_feature.names.index(self.target_class)
                    self.dataset = self.dataset.filter(lambda x: x['label'] == target_idx)
            
            # Get image feature
            if 'image' in self.dataset.features:
                self.images = self.dataset['image']
                # Generate image paths with indices and .jpg extension
                self.image_paths = [f"hf_dataset_{i}.jpg" for i in range(len(self.images))]
            else:
                raise ValueError("Dataset does not contain 'image' feature")
            
            # Get class names if available
            if 'label' in self.dataset.features:
                label_feature = self.dataset.features['label']
                if isinstance(label_feature, datasets.ClassLabel):
                    self.class_indices = label_feature.names
                
        except Exception as e:
            raise ValueError(f"Error loading HuggingFace dataset: {str(e)}")

    def __getitem__(self, idx: int):
        """Get a single sample"""
        try:
            if not self.is_local_dataset:
                # HuggingFace dataset
                img = self.images[idx]
                if isinstance(img, Image.Image):
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                elif isinstance(img, np.ndarray):
                    if img.shape[-1] != 3:
                        img = Image.fromarray(img).convert('RGB')
                img_path = self.image_paths[idx]
            else:
                # Local dataset
                img_path = self.images[idx]
                img = ImageDatasets.read_image(img_path)
            
            tensor = self.transforms(img)
            return img, tensor, img_path
            
        except Exception as e:
            print(f"Error loading image at index {idx}: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        """Collate function for DataLoader"""
        images, tensors, image_paths = tuple(zip(*batch))
        return images, torch.stack(tensors, dim=0), image_paths

    def get_class_indices(self):
        return self.class_indices

class CBIRDatasets(Dataset):
    def __init__(self, 
                 root: str, 
                 transforms = None,
                 postfix: tuple = ('jpg', 'png'),
                 mode: str = 'query'):

        assert transforms is not None, 'transforms would not be None'
        assert mode in ('query', 'gallery'), 'make sure mode is query or gallery'
        
        self.mode = mode
        self.transforms = transforms
        self.is_local_dataset = True
        
        try:
            # Try local dataset first
            query_dir, gallery_dir = os.path.join(opj(root, 'query')), os.path.join(opj(root, 'gallery'))
            assert os.path.isdir(query_dir) and os.path.isdir(gallery_dir), 'make sure query dir and gallery dir exists'
            self._init_from_local(postfix, query_dir, gallery_dir)
        except (AssertionError, ValueError):
            try:
                # Try HuggingFace dataset if local fails
                self._init_from_huggingface(root)
                self.is_local_dataset = False
            except Exception as e:
                raise ValueError(f"Failed to load dataset from local path or Hugging Face. Error: {str(e)}")

    def _init_from_local(self, postfix, query_dir, gallery_dir):
        """Initialize from local directory structure"""
        is_subset, query_identity, _ = self._check_subset(query_dir, gallery_dir) 
        if not is_subset:
            raise ValueError('query identity is not subset of gallery identity')

        data = {'query': [], 'pos': []}
        gallery = {'gallery': []}
        if self.mode == 'query':
            for q in query_identity:
                one_identity_queries = glob.glob(opj(query_dir, q, f'*.{postfix[0]}')) + glob.glob(opj(query_dir, q, f'*.{postfix[1]}'))
                one_identity_positives = glob.glob(opj(gallery_dir, q, f'*.{postfix[0]}')) + glob.glob(opj(gallery_dir, q, f'*.{postfix[1]}'))
                for one_q in one_identity_queries:
                    data['query'].append(one_q)
                    data['pos'].append(one_identity_positives)
        else:
            gallery['gallery'] = glob.glob(opj(gallery_dir, f'**/*.{postfix[0]}')) + glob.glob(opj(gallery_dir, f'**/*.{postfix[1]}'))
        
        self.data = datasets.Dataset.from_dict(data)
        self.gallery = datasets.Dataset.from_dict(gallery)

    def _init_from_huggingface(self, dataset_name):
        """Initialize from HuggingFace dataset"""
        dataset = load_dataset(dataset_name)

        # Verify dataset structure
        required_splits = ['query', 'gallery']
        if not all(split in dataset for split in required_splits):
            raise ValueError(f"Dataset must contain both 'query' and 'gallery' splits")
        
        if self.mode == 'query':
            # Convert HuggingFace format to match local format
            data = {'query': [], 'pos': []}
            query_data = dataset['query']
            gallery_data = dataset['gallery']
            
            # Group gallery images by class
            gallery_by_class = defaultdict(list)
            for item in gallery_data:
                gallery_by_class[item['class_name']].append(item['image'])
            
            # Create query-positive pairs
            for item in query_data:
                data['query'].append(item['image'])
                data['pos'].append(gallery_by_class[item['class_name']])
            
            self.data = datasets.Dataset.from_dict(data)
            self.gallery = datasets.Dataset.from_dict({'gallery': []})  # query mode is empty
        else:
            # gallery mode, only store all gallery images
            gallery_data = dataset['gallery']
            gallery = {'gallery': [item['image'] for item in gallery_data]}
            self.data = datasets.Dataset.from_dict({'query': [], 'pos': []})  # gallery mode is empty
            self.gallery = datasets.Dataset.from_dict(gallery)

    def __getitem__(self, idx: int):
        if self.mode == 'query':
            data = self.data[idx]['query']
        else:
            data = self.gallery[idx]['gallery']
            
        if self.is_local_dataset:
            data_image = ImageDatasets.read_image(data)
        else:
            # Handle HuggingFace image format
            if isinstance(data, Image.Image):
                data_image = data if data.mode == 'RGB' else data.convert('RGB')
            elif isinstance(data, np.ndarray):
                data_image = Image.fromarray(data).convert('RGB')
            else:
                raise ValueError(f"Unexpected image type: {type(data)}")

        tensor = self.transforms(data_image)
        return tensor

    @classmethod
    def build(cls, root: str, transforms = None, postfix: tuple = ('jpg', 'png')):
        return cls(root, transforms, postfix, 'query'), cls(root, transforms, postfix, 'gallery')

    def _check_subset(self, query: str, gallery: str):
        query_identity = [q for q in os.listdir(query) if not q.startswith('.')]
        gallery_identity = [q for q in os.listdir(gallery) if not q.startswith('.')]
        return set(query_identity).issubset(set(gallery_identity)), query_identity, gallery_identity
    
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