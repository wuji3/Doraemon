from .backbone.backbone_def import BackboneFactory
from .head.head_def import HeadFactory
import torch.nn as nn
from torch.nn.init import normal_, constant_
import torch
from typing import Union
import torch.nn.functional as F
import os
import numpy as np

class FaceTrainingWrapper:
    def __init__(self, model_cfg, logger = None):
        self.model = FaceTrainingModel(model_cfg)
        self.logger = logger

    def init_parameters(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            normal_(m.weight, mean=0, std=0.02)
        elif isinstance(m, nn.BatchNorm2d):
            constant_(m.weight, 1)
            constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            normal_(m.weight, mean=0, std=0.02)
            constant_(m.bias, 0)

    def reset_parameters(self):
        self.model.apply(self.init_parameters)

class FaceTrainingModel(nn.Module):
    """Define a traditional faceX model which contains a backbone and a head.

    Attributes:
        backbone: the backbone of faceX model.
        head: the head of faceX model.
    """

    def __init__(self, model_cfg):
        """Init faceX model by backbone factorcy and head factory.

        Args:
            backbone_factory: produce a backbone according to config files.
            head_factory: produce a head according to config files.
        """
        super().__init__()
        backbone = BackboneFactory(model_cfg['backbone']).get_backbone()
        head = HeadFactory(model_cfg['head']).get_head()
        self.trainingwrapper = nn.ModuleDict({
            'backbone': backbone,
            'head': head
        })

    def forward(self, data, label):
        feat = self.trainingwrapper['backbone'](data)
        pred = self.trainingwrapper['head'](feat, label)
        return pred

class FaceModelLoader:
    def __init__(self, model_cfg: dict):
        self.model = BackboneFactory(model_cfg['backbone']).get_backbone()

    def load_weight_default(self, model_path):
        """The default method to load a model.

        Args:
            model_path:: the path of the weight file.

        Returns:
            model: initialized model.
        """
        self.model.load_state_dict(torch.load(model_path)['state_dict'], strict=True)

        return self.model

    def load_weight(self, model_path, ema: bool = False):
        """The custom method to load a model, from a model having feature extractor and head.

        Args:
            model_path: the path of the weight file.

        Returns:
            model: initialized model.
        """
        model_dict = self.model.state_dict()

        pretrained_dict = torch.load(model_path)['ema'].float().state_dict() if ema else torch.load(model_path)['state_dict']

        new_pretrained_dict = {}
        for k in model_dict:
            #new_pretrained_dict[k] = pretrained_dict['trainingwrapper.backbone.' + k]  # tradition training
            new_pretrained_dict[k] = pretrained_dict['backbone.' + k]  # tradition training

        model_dict.update(new_pretrained_dict)
        self.model.load_state_dict(model_dict)

        return self.model

class FeatureExtractor:

    def __init__(self, model):
        self.model = model

    def extract_face(self, dataloader, device) -> dict:
        """Extract and return features.

        Args:
            model: initialized model.
            dataloader: load data to be extracted.

        Returns:
            image_name2feature: key is the name of image, value is feature of image.
        """
        model = self.model
        model.eval()
        model.to(device)

        image_name2feature = {}
        with torch.no_grad():
            for batch_idx, (_, tensors, file_realpaths) in enumerate(dataloader):
                tensors = tensors.to(device)
                features = model(tensors)
                features = F.normalize(features)
                features = features.cpu().numpy()
                for realpath, feature in zip(file_realpaths, features):
                    filename = os.path.join(os.path.basename(os.path.dirname(realpath)), os.path.basename(realpath))
                    image_name2feature[filename] = feature

        return image_name2feature
    
    def extract_cbir(self, dataloader, device) -> dict:
        """Extract and return features.

        Args:
            model: initialized model.
            dataloader: load data to be extracted.

        Returns:
            features: feature of image.
        """
        model = self.model
        model.eval()
        model.to(device)

        features = []
        with torch.no_grad():
            for batch_idx, tensors in enumerate(dataloader):
                tensors = tensors.to(device)
                feature = model(tensors)
                feature = F.normalize(feature)
                feature = feature.cpu().numpy()

                features.append(feature)

        return np.concatenate(features, axis=0)