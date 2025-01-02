import torchvision
from copy import deepcopy
from torch.nn.init import normal_, constant_
import torch.nn as nn
from built.attention_based_pooler import atten_pool_replace
import torch
from .faceX import FaceTrainingWrapper
import torch.distributed as dist
import timm  

class SetOutFeatures:

    def __init__(self):
        self.models = {
            'mobilenet',  # mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small
            'shufflenet',  # shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50', 'resnext101', # res-series
            'convnext',  # convnext_tiny, convnext_small, convnext_base, convnext_large
            'efficientnet',  # efficientnet_b0 -> efficientnet_b7, efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l
            'swin',
        }

    def init_nc(self, model: nn.Module, choice: str, nc: int, model_type: str = 'torchvision') -> None:
        if model_type == 'timm':
            # timm models typically use 'head' or 'classifier' as final layer
            if hasattr(model, 'head'):
                if isinstance(model.head, nn.Linear):
                    model.head = nn.Linear(model.head.in_features, nc)
            elif hasattr(model, 'classifier'):
                if isinstance(model.classifier, nn.Linear):
                    model.classifier = nn.Linear(model.classifier.in_features, nc)
            return

        model_name = choice.split('_')[0]
        assert model_name in self.models, 'Not supported model'
        if model_name in {'mobilenet', 'convnext', 'efficientnet'}:
            m = getattr(model, 'classifier')
            if isinstance(m, nn.Sequential):
                m[-1] = nn.Linear(m[-1].in_features, nc)
        elif model_name == 'swin':
            model.head = nn.Linear(model.head.in_features, nc)
        else:
            model.fc = nn.Linear(model.fc.in_features, nc)

class TorchVisionWrapper:
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
        model_cfgs_copy['kind'], model_cfgs_copy['choice'] = model_cfgs['name'].split('-')

        # 修改这里以支持 timm
        self.init_nc_helper = SetOutFeatures()
        self.model = self.create_model(**model_cfgs_copy)
        # pool layer
        if model_cfgs['attention_pool']:
            self.model = atten_pool_replace(self.model)

        del model_cfgs_copy

        if not self.pretrained: self.reset_parameters()

    def create_model(self, choice: str, num_classes: int = 1000, pretrained: bool = False, 
                    kind: str = 'torchvision', backbone_freeze: bool = False, 
                    bn_freeze: bool = False, bn_freeze_affine: bool = False, 
                    load_from: str = None, **kwargs):
        assert kind in {'torchvision', 'timm', 'custom'}, 'kind must be torchvision, timm or custom'
        
        if kind == 'timm':
            # 创建 timm 模型
            model = timm.create_model(
                choice,
                pretrained=pretrained,
                num_classes=num_classes,
                **kwargs['kwargs']
            )
            
            # 处理自定义权重加载
            if load_from is not None:
                weights = self.load_weight(load_from)
                weights_out_nc = weights[list(weights.keys())[-1]].shape[0]
                self.init_nc_helper.init_nc(model, choice, weights_out_nc, kind)
                model.load_state_dict(weights)
                if weights_out_nc != num_classes:
                    self.init_nc_helper.init_nc(model, choice, num_classes, kind)
                if self.logger is not None and self.rank in (-1, 0):
                    self.logger.both(f'load_from: {load_from}')
            
        elif kind == 'torchvision':
            # Get weights using get_model_weights
            weights = torchvision.models.get_model_weights(choice).DEFAULT if pretrained else None
            
            if pretrained and self.rank == 0:
                torchvision.models.get_model(choice, weights=weights, **kwargs['kwargs'])
            if self.rank >= 0: dist.barrier()

            model = torchvision.models.get_model(choice, weights=weights, **kwargs['kwargs'])
            
            # load weight if need
            if load_from is not None:
                weights = self.load_weight(load_from)
                weigths_out_nc = weights[list(weights.keys())[-1]].numel()
                self.init_nc_helper.init_nc(model, choice, weigths_out_nc)
                model.load_state_dict(weights)
                if weigths_out_nc != num_classes: 
                    self.init_nc_helper.init_nc(model, choice, num_classes)
                if self.logger is not None and self.rank in (-1, 0): 
                    self.logger.both(f'load_from: {load_from}')
            else:
                # init num_classes from torchvision.models
                self.init_nc_helper.init_nc(model, choice, num_classes)

        else:
            model = ...
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
        kind, _ = self.model_cfgs['choice'].split('-')
        if kind == 'torchvision':
            for name, m in self.model.named_children():
                if name not in ('fc', 'classifier', 'head'):
                    for p in m.parameters():
                        p.requires_grad_(False)
            print('backbone freeze')
        else: pass

    def freeze_bn(self, bn_freeze_affine: bool = False):
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                if bn_freeze_affine:
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

def get_model(model_cfg, logger, rank):
    assert 'task' in model_cfg, 'image classification or face recognition ?'

    match model_cfg['task']:
        case 'face' | 'cbir': return FaceTrainingWrapper(model_cfg, logger)
        case 'classification': return TorchVisionWrapper(model_cfg, logger, rank)