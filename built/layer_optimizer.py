from utils.optimizer import BaseSeperateLayer
from typing import Iterator, List, Dict, Union, Optional
from torch.nn import Module

class SeperateLayerParams(BaseSeperateLayer):
    def __init__(self, model: Module):
        super().__init__(model)

    def create_ParamSequence(self, alpha: Optional[float], lr: float) -> Union[Iterator, List[Dict]]:
        """
        torchvision-swin_t
        Args:
            alpha: lr衰减系数
            lr: 基准学习率

        Returns:
            params: torch.optim.Optimizer中的params
        """
        if alpha is None: return self.model.parameters()

        params = [
            {'params': self.model.features.parameters()},
            {'params': self.model.norm.parameters()},
            {'params': self.model.head.parameters(), 'lr': lr * alpha}
        ]
        return params