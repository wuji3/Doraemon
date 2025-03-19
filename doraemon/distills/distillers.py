import torch.nn as nn
from torch import Tensor
from typing import Callable
from ..engine.vision_engine import CenterProcessor

class Distiller:
    def __init__(self,
                 model_teacher: nn.Module,
                 model_student: nn.Module,
                 criterion_cls: Callable,
                 criterion_kl: Callable,
                 cls_weight: float = 0.5,
                 kl_weight: float = 0.5):

        self.model_teacher = model_teacher
        self.model_student = model_student
        self.criterion_cls = criterion_cls
        self.criterion_kl = criterion_kl
        self.cls_weight = cls_weight
        self.kl_weight = kl_weight

    def __call__(self, inputs: Tensor, label: Tensor) -> Tensor:
        # forward
        logit_s = self.model_student(inputs)
        logit_t = self.model_teacher(inputs)
        # compute loss
        loss_cls = self.criterion_cls(logit_s, label)
        loss_kl = self.criterion_kl(logit_s, logit_t)
        loss = self.cls_weight * loss_cls + self.kl_weight * loss_kl

        return loss

class DistillCenterProcessor(CenterProcessor):
    def __init__(self, cfgs: dict, rank: int, project: str, logger = None, opt = None):
        super().__init__(cfgs=cfgs['student'], rank=rank, project= project)

        # init teacher model
        # self.teacher = TorchVisionWrapper(cfgs['teacher'], logger = logger)
        # self.opt = opt




























