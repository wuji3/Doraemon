import torch
import torch.nn as nn
from typing import Callable
from functools import wraps

__all__ = ['bce',
           'ce',
           'focal',
           'create_Lossfn',
           'list_lossfns',]

LOSS = {}

def register_loss(fn: Callable):
    key = fn.__name__
    if key in LOSS:
        raise ValueError(f"An entry is already registered under the name '{key}'.")
    LOSS[key] = fn
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha= 0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

@register_loss
def bce():
    return nn.BCEWithLogitsLoss()
@register_loss
def ce(label_smooth: float = 0.):
    return nn.CrossEntropyLoss(label_smoothing=label_smooth)
@register_loss
def focal():
    return FocalLoss(loss_fcn = nn.BCEWithLogitsLoss(), alpha=0.25)
def create_Lossfn(lossfn: str):
    lossfn = lossfn.strip()
    return LOSS[lossfn]

def list_lossfns():
    lossfns = [k for k, v in LOSS.items()]
    return sorted(lossfns)
