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
from ..built.class_augmenter import ClassWiseAugmenter
from prettytable import PrettyTable
import datasets
import numpy as np
from typing import Optional
import pandas as pd

class ImageDatasets(Dataset):
    def __init__(self, 
                 root_or_dataset, 
                 mode='train', 
                 transforms=None, 
                 label_transforms=None, 
                 project=None, 
                 rank=None, 
                 training=True,
                 id2label=None):

        self.transforms = transforms
        self.label_transforms = label_transforms
        self.is_local_dataset = True
        self.training = training
        self.id2label = id2label
        self.label2id = {label: i for i, label in id2label.items()} if id2label is not None else None

        if not training and id2label is not None:
            self.id2label = id2label
            self.label2id = {label: i for i, label in id2label.items()}

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
            self.id2label = {i: label for i, label in enumerate(data_class)}
            self.label2id = {label: i for i, label in self.id2label.items()}
            if rank in {-1, 0}:
                self._save_mappings(project)
        
        # Vectorize labels
        self.labels = df[data_class].values.tolist()  # shape: (num_samples, num_classes)
    
    def _init_from_local(self, root, mode, project, rank):
        assert os.path.isdir(root), f"Dataset root: {root} does not exist."
        src_path = os.path.join(root, mode)
        
        classes = [cla for cla in os.listdir(src_path) if os.path.isdir(os.path.join(src_path, cla))]
        classes.sort()

        if self.training:
            self.id2label = {i: label for i, label in enumerate(classes)}
            self.label2id = {label: i for i, label in self.id2label.items()}
            if rank in {-1, 0}:
                self._save_mappings(project)

        support = [".jpg", ".png", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"]

        images_path = []
        images_label = []

        for cla in classes:
            cla_path = os.path.join(src_path, cla)
            images = [os.path.join(src_path, cla, i) for i in os.listdir(cla_path)
                      if os.path.splitext(i)[-1].lower() in support]
            class_id = self.label2id[cla]
            images_path.extend(images)
            images_label += [class_id] * len(images)

        self.images = images_path
        self.labels = images_label

    def _init_from_huggingface(self, dataset_name, split, project, rank):
        if split.startswith("val"):
            split = "validation"
        self.dataset = load_dataset(dataset_name, split=split)
        
        if 'label' not in self.dataset.features:
            raise ValueError("Dataset does not contain 'label' feature")

        label_feature = self.dataset.features['label']
        if not (isinstance(label_feature, datasets.ClassLabel) or isinstance(label_feature, datasets.Value)):
            raise ValueError("Dataset.features['label'] is not ClassLabel OR Value type")

        if self.training:
            if isinstance(label_feature, datasets.ClassLabel):
                classes = label_feature.names
                self.id2label = {i: label for i, label in enumerate(classes)}
                self.label2id = {label: i for i, label in self.id2label.items()}
            elif isinstance(label_feature, datasets.Value):
                self.id2label = {i: label for i, label in zip(self.dataset['label'], self.dataset['class_name'])}
                self.label2id = {label: i for i, label in self.id2label.items()}
            if rank in {-1, 0}:
                self._save_mappings(project)

        # self.images = self.dataset['image']
        # self.labels = self.dataset['label']

    def _save_mappings(self, project):
        if project is not None:
            id2label_path = Path(project) / 'id2label.json'
            if not id2label_path.exists():
                os.makedirs(id2label_path.parent, exist_ok=True)
                with open(id2label_path, 'w') as f:
                    json.dump(self.id2label, f, indent=4)

    def __getitem__(self, idx):
        if hasattr(self, 'dataset'):  # Hugging Face dataset
            try:
                img = self.dataset[idx]['image']
                label = self.dataset[idx]['label']
                img = ImageDatasets.load_image_from_hf(img)
            except Exception as e:
                random_idx = np.random.randint(0, len(self.dataset))
                while random_idx == idx:
                    random_idx = np.random.randint(0, len(self.dataset))
                return self.__getitem__(random_idx)
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
            img = self.transforms(img, label, self.id2label) if type(self.transforms) is ClassWiseAugmenter else self.transforms(img)

        if self.label_transforms is not None:
            label = self.label_transforms(label)

        return img, label

    def __len__(self):
        return self.dataset.shape[0] if hasattr(self, 'dataset') else len(self.images)

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
    def tell_data_distribution(datasets, logger, nc: int):
        """
        Display the distribution of samples across classes for both train and val sets.
        
        Args:
            datasets: Dictionary containing train and val datasets
            logger: Logger instance for output
            nc: Number of classes
        """
        data_distribution = defaultdict(lambda: {'train': 0, 'val': 0})
        
        for split, dataset in datasets.items():
            if hasattr(dataset, 'multi_label') and dataset.multi_label:
                # For multi-label data from CSV
                for label_vector in dataset.labels:
                    # label_vector is a list of 0/1 indicating presence of each class
                    for idx, is_present in enumerate(label_vector):
                        if is_present > 0:
                            class_name = dataset.id2label[idx]
                            data_distribution[class_name][split] += 1
            else:
                # For single-label data
                if hasattr(dataset, 'labels'): # For local dataset
                    for label in dataset.labels:
                        class_name = dataset.id2label[label]
                        data_distribution[class_name][split] += 1
                else: # For HuggingFace dataset
                    for label in dataset.dataset['label']:
                        class_name = dataset.id2label[label]
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
        pretty_table.add_row(['TOTAL', train_total, val_total])

        # Output the table
        msg = '\n' + str(pretty_table)
        logger.both(msg) if nc <= 50 else logger.log(msg)
        return [(class_name, data_distribution[class_name]) for class_name in sorted_classes]


class PredictImageDatasets(Dataset):
    def __init__(self, root=None, transforms=None, postfix: tuple=('jpg', 'png'), 
                 sampling: Optional[int]=None, id2label: Optional[list]=None,
                 classes: Optional[list]=None, split: Optional[str] = None, require_gt: bool = False):
        """
        Dataset for prediction, supporting directory, CSV file, and HuggingFace dataset inputs
        
        Args:
            root: Path to image directory, CSV file, or HuggingFace dataset name
            transforms: Image transformations
            postfix: Tuple of image extensions (for directory mode)
            sampling: Number of samples to use (for subset testing)
            id2label: List of class names
            classes: Filter dataset to only include specific class
            split: Split to filter dataset
            require_gt: Boolean indicating if ground truth labels are required
        """
        assert transforms is not None, 'transforms would not be None'

        self.transforms = transforms
        self.id2label = id2label
        self.is_local_dataset = True
        self.multi_label = False
        self.classes = classes
        self.split = split
        self.require_gt = require_gt
        self.gt_labels = []

        if root is None:  # used for face embedding infer
            self.images = []
        else:
            try:
                if os.path.isfile(root) and root.endswith('.csv'):
                    self.multi_label = True
                    self._init_from_csv(root, split=split)
                else:
                    self._init_from_dir(root, postfix, split=split)
            except (ValueError, FileNotFoundError):
                try:
                    self._init_from_huggingface(root, split=split)
                    self.is_local_dataset = False
                except Exception as e:
                    raise ValueError(f"Failed to load dataset from {root}. Error: {str(e)}")

        if sampling is not None:
            self.images = self.images[:sampling]

    def _init_from_csv(self, csv_path, split=None):
        df = pd.read_csv(csv_path)
        
        # Filter split
        if split is not None:
            if 'train' not in df.columns:
                raise ValueError("CSV must contain 'train' column for split filtering")
            df = df[df['train'] == (split == 'train')]

        assert 'image_path' in df.columns, 'CSV must contain image_path column'
        
        if self.require_gt:
            # default task：ground truth needed
            available_classes = [col for col in df.columns if col not in ['image_path', 'train']]
            if not available_classes:
                raise ValueError("GT labels required but no label columns found in CSV")
            
            if self.classes is not None:
                target_classes = self.classes if isinstance(self.classes, list) else [self.classes]
                # verify target classes
                for class_name in target_classes:
                    assert class_name in available_classes, f'Target class {class_name} not found in CSV columns'
            else:
                target_classes = available_classes

            # filter samples by classes
            if self.classes is not None:
                mask = df[target_classes].any(axis=1)
                df = df[mask].reset_index(drop=True)

            # process labels
            for _, row in df.iterrows():
                sample_labels = [c for c in target_classes if row[c] > 0.5]
                self.gt_labels.append(sample_labels if sample_labels else None)
        else:
            # autolabel task: only image paths
            self.gt_labels = [None] * len(df)
            
        self.images = df['image_path'].tolist()
        assert len(self.images) > 0, 'No valid image paths found in CSV'

    def _init_from_dir(self, root, postfix, split=None):
        if not os.path.isdir(root):
            raise ValueError(f"The provided path {root} is not a directory")

        # 处理split
        if split is not None:
            split_dir = os.path.join(root, split)
            if not os.path.isdir(split_dir):
                raise ValueError(f"Split directory not found: {split_dir}")
            root = split_dir

        self.images = []

        if self.require_gt:
            # default task: get labels from directory structure
            available_classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            if not available_classes:
                raise ValueError("GT labels required but no class directories found")
            
            if self.classes is not None:
                target_classes = self.classes if isinstance(self.classes, list) else [self.classes]
                # verify target classes
                for class_name in target_classes:
                    if class_name not in available_classes:
                        raise ValueError(f"Target class {class_name} not found in directory")
            else:
                target_classes = available_classes

            # iterate target classes
            for class_name in target_classes:
                class_dir = os.path.join(root, class_name)
                for ext in postfix:
                    class_images = glob.glob(os.path.join(class_dir, f'*.{ext}'))
                    self.images.extend(class_images)
                    self.gt_labels.extend([class_name] * len(class_images))
        else:
            # autolabel task: flat directory structure, get all images
            for ext in postfix:
                self.images.extend(glob.glob(os.path.join(root, f'**/*.{ext}'), recursive=True))
            self.gt_labels = [None] * len(self.images)

        assert len(self.images) > 0, f'No files found with postfix {postfix}'

    def _init_from_huggingface(self, dataset_name, split=None):
        """Initialize dataset from HuggingFace"""
        try:
            from datasets import load_dataset
            # Load validation split by default for prediction
            if split is None:
                dataset = load_dataset(dataset_name)
                all_splits = list(dataset.keys())
                if len(all_splits) > 1:
                    self.dataset = datasets.concatenate_datasets([dataset[s] for s in all_splits])
                else:
                    self.dataset = dataset[all_splits[0]]
            elif split.startswith("val") or split.startswith("test"):
                self.dataset = load_dataset(dataset_name, split='validation')
            else:
                self.dataset = load_dataset(dataset_name, split=split)
            
            # Filter by target classes if specified
            if self.classes is not None:
                if 'label' not in self.dataset.features:
                    raise ValueError("Dataset does not contain 'label' feature")
                
                classes = self.classes if isinstance(self.classes, list) else [self.classes]
                label_feature = self.dataset.features['label']
                if isinstance(label_feature, datasets.ClassLabel):
                    breakpoint()
                    # Get indices of all target classes
                    target_indices = []
                    for class_name in classes:
                        if class_name not in label_feature.names:
                            raise ValueError(f"Target class {class_name} not found in dataset classes")
                        target_indices.append(label_feature.names.index(class_name))
                    # Filter dataset to keep samples from any of the target classes
                    self.dataset = self.dataset.filter(lambda x: x['label'] in target_indices)
            
            # Get image feature
            if 'image' in self.dataset.features:
                self.images = self.dataset['image']
                # Generate image paths with indices and .jpg extension
                self.image_paths = [f"hf_dataset_{i}.jpg" for i in range(len(self.images))]
            else:
                raise ValueError("Dataset does not contain 'image' feature")
            
            # Initialize gt_labels based on requirements
            if self.require_gt and 'label' in self.dataset.features:
                label_feature = self.dataset.features['label']
                if isinstance(label_feature, datasets.ClassLabel):
                    self.id2label = label_feature.names
                    self.gt_labels = [label_feature.names[label] for label in self.dataset['label']]
                else:
                    self.gt_labels = [None] * len(self.images)
            else:
                self.gt_labels = [None] * len(self.images)
            
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
            gt_label = self.gt_labels[idx] if self.require_gt else None
            return img, tensor, img_path, gt_label
            
        except Exception as e:
            print(f"Error loading image at index {idx}: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        """Collate function for DataLoader"""
        images, tensors, image_paths, gt_labels = tuple(zip(*batch))
        return images, torch.stack(tensors, dim=0), image_paths, gt_labels

    def get_id2label(self):
        return self.id2label

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
    
        # Check if query identities are subset of gallery identities
        query_identity = set(dataset['query']['class_name'])
        gallery_identity = set(dataset['gallery']['class_name'])
        if not query_identity.issubset(gallery_identity):
            raise ValueError('query identity is not subset of gallery identity')

        self.dataset = dataset

        if self.mode == 'query':
            self.query_indices = {
                filename: idx 
            for idx, filename in enumerate(dataset['query']['file_name'])
            }
            self.gallery_indices = {}

            data = {
                'query': [],  # Will store file_name
                'pos': []     # Will store lists of file_names
            }

            # Group gallery images by class_name
            gallery_by_class = defaultdict(list)
            for item in dataset['gallery']:
                gallery_by_class[item['class_name']].append(item['file_name'])

            # Create query-positive pairs using file_name
            for item in dataset['query']:
                query_file = item['file_name']
                class_name = item['class_name']
                positive_files = gallery_by_class[class_name]

                data['query'].append(query_file)
                data['pos'].append(positive_files)

            self.data = datasets.Dataset.from_dict(data)
            self.gallery = None

        else:  # gallery mode
            self.gallery_indices = {
                filename: idx 
            for idx, filename in enumerate(dataset['gallery']['file_name'])
            }
            self.query_indices = {}

            gallery = {'gallery': [item['file_name'] for item in dataset['gallery']]}
            self.gallery = datasets.Dataset.from_dict(gallery)
            self.data = None

    def __getitem__(self, idx: int):
        if self.mode == 'query':
            file_name = self.data[idx]['query']
        else:
            file_name = self.gallery[idx]['gallery']

        data_image = self.get_image(file_name)
        tensor = self.transforms(data_image)

        return tensor
    
    def get_image(self, file_name):
        """Access image by file name, support local and HuggingFace dataset"""
        if self.is_local_dataset:
            return ImageDatasets.read_image(file_name)
        else:
            # find image by file name
            if file_name in self.query_indices:
                raw_image = self.dataset['query'][self.query_indices[file_name]]['image']
                return ImageDatasets.load_image_from_hf(raw_image)
            elif file_name in self.gallery_indices:
                raw_image = self.dataset['gallery'][self.gallery_indices[file_name]]['image']
                return ImageDatasets.load_image_from_hf(raw_image)
            else:
                raise ValueError(f"Image {file_name} not found in dataset")

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