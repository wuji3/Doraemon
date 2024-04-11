from .backbone.backbone_def import BackboneFactory
from .head.head_def import HeadFactory
import torch.nn as nn
from torch.nn.init import normal_, constant_
import torch
from typing import Optional

class FaceWrapper:
    def __init__(self, model_cfg, logger = None):
        self.model = FaceModel(model_cfg)
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

class FaceModel(nn.Module):
    """Define a traditional faceX model which contains a backbone and a head.

    Attributes:
        backbone(object): the backbone of faceX model.
        head(object): the head of faceX model.
    """

    def __init__(self, model_cfg):
        """Init faceX model by backbone factorcy and head factory.

        Args:
            backbone_factory(object): produce a backbone according to config files.
            head_factory(object): produce a head according to config files.
        """
        super().__init__()
        backbone = BackboneFactory(model_cfg['backbone']).get_backbone()
        head = HeadFactory(model_cfg['head']).get_head()
        self.model = nn.ModuleList([backbone, head])

    def forward(self, data, label):
        feat = self.model[0](data)
        pred = self.model[1](feat, label)
        return pred

class FaceFeatureModel:
    def __init__(self, model_cfg: dict, model_path: Optional[str] = None):
        self.model = BackboneFactory(model_cfg['backbone']).get_backbone()
        if model_path is not None:
            self.load_model(model_path=model_path)

    def load_model_default(self, model_path):
        """The default method to load a model.

        Args:
            model_path:: the path of the weight file.

        Returns:
            model: initialized model.
        """
        self.model.load_state_dict(torch.load(model_path)['state_dict'], strict=True)

        return self.model

    def load_model(self, model_path):
        """The custom method to load a model.

        Args:
            model_path: the path of the weight file.

        Returns:
            model: initialized model.
        """
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(model_path)['state_dict']

        new_pretrained_dict = {}
        for k in model_dict:
            new_pretrained_dict[k] = pretrained_dict['backbone.' + k]  # tradition training

        model_dict.update(new_pretrained_dict)
        self.model.load_state_dict(model_dict)

        return self.model

