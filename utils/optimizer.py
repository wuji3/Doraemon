from torch.optim import SGD, Adam
from typing import Callable, List, Dict, Optional, Iterator, Union
from functools import wraps
from abc import ABCMeta, abstractmethod
from torch.nn import Module

__all__ = ['SGD',
           'Adam',
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

def create_Optimizer(optimizer: str, lr: float, weight_decay, momentum, params):
    # return partial(OPTIMIZER[optimizer], lr = lr, weight_decay = weight_decay, momentum = momentum)
    return OPTIMIZER[optimizer](params = params, lr = lr, weight_decay = weight_decay, momentum = momentum)

def list_optimizers():
    optimizers = [k for k, v in OPTIMIZER.items()]
    return sorted(optimizers)