from torch.optim import SGD, Adam
from typing import Callable, List, Dict, Optional, Iterator, Union
from functools import wraps
from abc import ABCMeta, abstractmethod
from torch.nn import Module
import torch
from torch.nn.modules.batchnorm import _BatchNorm

__all__ = ['sgd',
           'adam',
           'sam',
           'BaseSeperateLayer',
           'create_Optimizer',
           'list_optimizers']

OPTIMIZER = {}

def register_optimizer(fn: Callable):
    key = fn.__name__
    if key in OPTIMIZER:
        raise ValueError(f"An entry is already registered under the name '{key}'.")
    OPTIMIZER[key] = fn
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper

class SAM(torch.optim.Optimizer):
    """
    https://arxiv.org/abs/2010.01412 Sharpness-Aware Minimization for Efficiently Improving Generalization
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=True, lr: float = 0.01, momentum: float = 0.9, weight_decay: float = 5e-4, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def disable_running_stats(self, model):
        def _disable(module):
            if isinstance(module, _BatchNorm):
                module.backup_momentum = module.momentum
                module.momentum = 0

        model.apply(_disable)

    def enable_running_stats(self, model):
        def _enable(module):
            if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
                module.momentum = module.backup_momentum

        model.apply(_enable)

class BaseSeperateLayer(metaclass=ABCMeta):
    """
        用于对model的多个层分别设置具体的学习率
    """
    def __init__(self, model: Module) -> None:
        self.model = model

    @abstractmethod
    def create_ParamSequence(self, alpha: Optional[float], lr: float) -> Union[Iterator, List[Dict]]:
        pass

@register_optimizer
def sgd(*args, **kwargs):
    return SGD(*args, **kwargs)

@register_optimizer
def adam(*args, **kwargs):
    return Adam(*args, **kwargs)

@register_optimizer
def sam(base_optimizer = SGD, *args, **kwargs):
    return SAM(base_optimizer=base_optimizer, *args, **kwargs)

def create_Optimizer(optimizer: str, lr: float, weight_decay, momentum, params):
    # return partial(OPTIMIZER[optimizer], lr = lr, weight_decay = weight_decay, momentum = momentum)
    return OPTIMIZER[optimizer](params = params, lr = lr, weight_decay = weight_decay, momentum = momentum)

def list_optimizers():
    optimizers = [k for k, v in OPTIMIZER.items()]
    return sorted(optimizers)