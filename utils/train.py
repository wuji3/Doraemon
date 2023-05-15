import torch
from tqdm import tqdm
from typing import Callable
from functools import wraps
from .valuate import val
import math

__all__ = ['train_one_epoch']

STRATEGY = {}
def register_strategy(fn: Callable):
    key = fn.__name__.split('_')[-1]
    if key in STRATEGY:
        raise ValueError(f"An entry is already registered under the name '{key}'.")
    STRATEGY[key] = fn
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper

def make_divisible(x: int, divisor = 32):
    # Returns nearest x divisible by divisor
    return math.ceil(x / divisor) * divisor

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps
    # t = torch.full_like(torch.randn([3,5]), 0.05)
    # t[range(3), torch.tensor([1,0,4])] = 0.95

def print_imgsz(images: torch.Tensor):
    h, w = images.shape[-2:]
    return [h,w]

def mixup_data(x, y, device, lam):
    '''Returns mixed inputs, pairs of targets'''
    batch_size = x.size()[0]
    if device != torch.device('cpu'):
        index = torch.randperm(batch_size).to(device)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def update(model, loss, scaler, optimizer):
    # backward
    scaler.scale(loss).backward()

    # optimize
    scaler.unscale_(optimizer)  # unscale gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
    scaler.step(optimizer)
    scaler.update()

    optimizer.zero_grad()

def train_one_epoch(model, train_dataloader, val_dataloader, criterion, optimizer,
                    scaler, device: torch.device, epoch: int,
                    epochs: int, logger, is_mixup: bool, rank: int,
                    lam, schduler):
    # train mode
    model.train()

    cuda: bool = device != torch.device('cpu')

    if rank != -1:
        train_dataloader.sampler.set_epoch(epoch)
    pbar = enumerate(train_dataloader)
    if rank in {-1, 0}:
        pbar = tqdm(enumerate(train_dataloader),
                    total=len(train_dataloader),
                    bar_format='{l_bar}{bar:10}{r_bar}')

    tloss, fitness = 0., 0.,

    for i, (images, labels) in pbar:  # progress bar
        images, labels = images.to(device, non_blocking=True), labels.to(device)

        with torch.cuda.amp.autocast(enabled=cuda):
            # mixup
            if is_mixup:
                images, targets_a, targets_b = mixup_data(images, labels, device, lam)
                loss = mixup_criterion(criterion, model(images), targets_a, targets_b, lam)
            else:
                loss = criterion(model(images), labels)
        # scale + backward + grad_clip + step + zero_grad
        update(model, loss, scaler, optimizer)

        if rank in {-1, 0}:
            tloss = (tloss * i + loss.item()) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if cuda else 0)  # (GB)
            pbar.desc = f"{f'{epoch + 1}/{epochs}':>10}{mem:>10}{tloss:>12.3g}" + ' ' * 36
            pbar.postfix = f'lr:{optimizer.param_groups[0]["lr"]:.5f}, imgsz:{print_imgsz(images)}'

            if i == len(pbar) - 1:  # last batch
                logger.log(f'epoch:{epoch + 1:d}  t_loss:{tloss:4f}  lr:{optimizer.param_groups[0]["lr"]:.5f}')
                logger.log(f'{"name":<8}{"nums":>8}{"top1":>10}{"top5":>10}')

                # val
                top1, top5, v_loss = val(model, val_dataloader, device, pbar, True, criterion, logger)
                logger.log(f'v_loss:{v_loss:4f}  mtop1:{top1:.3g}  mtop5:{top5:.3g}\n')

                fitness = top1  # define fitness as top1 accuracy

    schduler.step()  # step epoch-wise

    return fitness

