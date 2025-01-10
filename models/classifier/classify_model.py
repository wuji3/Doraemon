from copy import deepcopy
import torch.nn as nn
from built.attention_based_pooler import atten_pool_replace
import torch
import torch.distributed as dist
import timm  
from torch.nn.init import normal_, constant_

class VisionWrapper:
    def __init__(self, model_cfgs: dict, logger = None, rank = -1):
        self.logger = logger
        self.model_cfgs = model_cfgs
        self.rank = rank

        self.kwargs = model_cfgs['kwargs']
        self.num_classes = model_cfgs['num_classes']
        self.pretrained = model_cfgs['pretrained']
        self.backbone_freeze = model_cfgs['backbone_freeze']
        self.bn_freeze = model_cfgs['bn_freeze']
        self.bn_freeze_affine = model_cfgs['bn_freeze_affine']

        model_cfgs_copy = deepcopy(model_cfgs)
        _, model_cfgs_copy['choice'] = model_cfgs['name'].split('-')

        self.model = self.create_model(**model_cfgs_copy)
        # pool layer
        if model_cfgs['attention_pool']:
            self.model = atten_pool_replace(self.model)

        del model_cfgs_copy

        if not self.pretrained: self.reset_parameters()

    def create_model(self, choice: str, num_classes: int = 1000, pretrained: bool = False, 
                    backbone_freeze: bool = False, bn_freeze: bool = False, 
                    bn_freeze_affine: bool = False, **kwargs):
        # Only rank 0 downloads the pre-trained weights
        if pretrained and self.rank == 0:
            _ = timm.create_model(
                choice,
                pretrained=True,
                num_classes=num_classes,
                **kwargs['kwargs']
            )

        if self.rank >= 0: 
            dist.barrier()
        
        model = timm.create_model(
            choice,
            pretrained=pretrained,
            num_classes=num_classes,
            **kwargs['kwargs']
        )

        if backbone_freeze: self.freeze_backbone()
        if bn_freeze: self.freeze_bn(bn_freeze_affine)

        return model

    def load_weight(self, load_from_path: str):
        weights = torch.load(load_from_path, map_location='cpu')
        weights = weights['ema'].float().state_dict() if weights.get('ema', None) is not None else weights['model']

        return weights

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

    def freeze_backbone(self):
        # Get the classifier module
        classifier = self.model.get_classifier()
        
        # Freeze all parameters except the classifier
        for _, m in self.model.named_modules():
            if m is not classifier:  # Skip the classifier module
                for p in m.parameters(recurse=False):
                    p.requires_grad_(False)
                    
        if self.rank <= 0:  # Only print on main process
            print('backbone freeze')

    def freeze_bn(self, bn_freeze_affine: bool = False):
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                if bn_freeze_affine:
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)