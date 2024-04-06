from .backbone.backbone_def import BackboneFactory
from .head.head_def import HeadFactory
import torch.nn as nn
from torch.nn.init import normal_, constant_

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