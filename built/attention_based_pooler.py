import torch
import torch.nn as nn
from torchvision.models import MobileNetV3, ResNet, ConvNeXt, EfficientNet, SwinTransformer


class AttentionPooling(nn.Module):
    """
    Augmenting Convolutional networks with attention-based aggregation: https://arxiv.org/abs/2112.13692
    """
    def __init__(self, in_dim):
        super().__init__()
        self.cls_vec = nn.Parameter(torch.randn(in_dim))
        self.fc = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        # x.view: B,C,H,W -> BxHxW,C
        weights = torch.matmul(x.reshape(-1, x.shape[1]), self.cls_vec)
        # weights.view: BxHxW,C -> B, HxW
        weights = self.softmax(weights.reshape(x.shape[0], -1))
        # x.view: B,C,H,W -> B,C,HxW
        # (B,C,HxW) @ (B,HxW,1) -> B,C,1
        x = torch.bmm(x.reshape(x.shape[0], x.shape[1], -1), weights.unsqueeze(-1)).squeeze(-1)
        x = x + self.cls_vec
        x = self.fc(x)
        x = x + self.cls_vec
        return x

def atten_pool_replace(model: nn.Module):
    #--------------------------------定制池化函数-----------------------------------#
    if type(model) is MobileNetV3:
        model.avgpool = AttentionPooling(in_dim=model.classifier[0].in_features)
    elif type(model) is ResNet:
        model.avgpool = AttentionPooling(in_dim=model.fc.in_features)
    elif type(model) is ConvNeXt:
        model.avgpool = AttentionPooling(in_dim=model.classifier[-1].in_features)
    elif type(model) is EfficientNet:
        model.avgpool = AttentionPooling(in_dim=model.classifier[-1].in_features)
    elif type(model) is SwinTransformer:
        model.avgpool = AttentionPooling(in_dim=model.head.in_features)
    else:
        raise KeyError(f'{type(model)} not support attention-based pool')

    return model
    #-----------------------------------------------------------------------------#