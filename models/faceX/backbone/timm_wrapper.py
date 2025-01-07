import timm
import torch
import torch.nn as nn
from copy import deepcopy

class TimmWrapper(nn.Module):
    """A wrapper for timm models that handles different model architectures uniformly."""
    
    def __init__(self, 
                 model_name: str,
                 feat_dim: int,
                 image_size: int,
                 pretrained: bool = True,
                 **kwargs):
        super().__init__()
        
        # 直接创建模型，让 timm 处理所有细节
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # 移除分类头
            global_pool='',  # 移除全局池化
        )
        
        # 获取输出通道数
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, image_size, image_size)
            output = self.model(dummy_input)
            if isinstance(output, tuple):
                output = output[0]  # 某些模型返回元组
            _, channels, h, w = output.shape  # B, C, H, W
        
        # 创建输出层
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Flatten(1),
            nn.Linear(channels * h * w, feat_dim),
            nn.BatchNorm1d(feat_dim)
        )
        
        # 现在可以安全地移除不需要的层
        if hasattr(self.model, 'head'):
            del self.model.head
        if hasattr(self.model, 'global_pool'):
            del self.model.global_pool
        if hasattr(self.model, 'fc'):
            del self.model.fc
        if hasattr(self.model, 'classifier'):
            del self.model.classifier

    def forward(self, x):
        # 使用 forward_features 而不是完整的 forward
        if hasattr(self.model, 'forward_features'):
            x = self.model.forward_features(x)
        else:
            x = self.model(x)
            
        if isinstance(x, tuple):
            x = x[0]
        x = self.output_layer(x)
        return x