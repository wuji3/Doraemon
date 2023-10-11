import torch
from tqdm import tqdm
from typing import Callable
from functools import wraps
from engine.procedure.evaluation import valuate
import math
from engine.optimizer import SAM

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
    # to device
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# scale + backward + grad_clip + step + zero_grad
def update(model, loss, scaler, optimizer, ema=None):
    # backward
    scaler.scale(loss).backward()

    # optimize
    scaler.unscale_(optimizer)  # unscale gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
    scaler.step(optimizer)
    scaler.update()

    optimizer.zero_grad()
    if ema:
        ema.update(model)


def update_sam(model: torch.nn.Module, inputs, targets, optimizer, lossfn, rank, ema=None, mixup=False, **kwargs):
    # first forward-backward step
    optimizer.enable_running_stats(model)
    if not mixup:
        loss = lossfn(model(inputs), targets)
    else:
        loss = mixup_criterion(lossfn, model(inputs), **kwargs)
    if rank >= 0: # 多卡训练
        with model.no_sync():
            loss.mean().backward()
    else:
        loss.mean().backward()
    optimizer.first_step(zero_grad=True)

    # second forward-backward step
    optimizer.disable_running_stats(model)
    if not mixup:
        lossfn(model(inputs), targets).mean().backward()
    else:
        mixup_criterion(lossfn, model(inputs), **kwargs).mean().backward()
    optimizer.second_step(zero_grad=True)

    if ema:
        ema.update(model)

    return loss

def train_one_epoch(model, train_dataloader, val_dataloader, criterion, optimizer,
                    scaler, device: torch.device, epoch: int,
                    epochs: int, logger, is_mixup: bool, rank: int,
                    lam, schduler, ema, sampler = None, thresh = 0):
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
        if sampler is not None: # OHEM-Softmax
            with torch.no_grad():
                valid = sampler.sample(model(images), labels)
                images, labels = images[valid], labels[valid]
        with torch.cuda.amp.autocast(enabled=cuda):
            # mixup
            if is_mixup:
                images, targets_a, targets_b = mixup_data(images, labels, device, lam)
                if type(optimizer) is SAM:
                    kwargs = dict(y_a=targets_a, y_b=targets_b, lam=lam)
                    loss = update_sam(model, images, labels, optimizer, criterion, rank, ema, mixup=True, **kwargs)
                else:
                    loss = mixup_criterion(criterion, model(images), targets_a, targets_b, lam)
                    update(model, loss, scaler, optimizer, ema)
            else:
                if type(optimizer) is SAM:
                    loss = update_sam(model, images, labels, optimizer, criterion, rank, ema, mixup=False)
                else:
                    loss = criterion(model(images), labels)
                    update(model, loss, scaler, optimizer, ema)

        if rank in {-1, 0}:
            tloss = (tloss * i + loss.item()) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if cuda else 0)  # (GB)
            pbar.desc = f"{f'{epoch + 1}/{epochs}':>10}{mem:>10}{tloss:>12.3g}" + ' ' * 36
            pbar.postfix = f'lr:{optimizer.param_groups[0]["lr"]:.5f}, imgsz:{print_imgsz(images)}'

            if i == len(pbar) - 1:  # last batch
                logger.log(f'epoch:{epoch + 1:d}  t_loss:{tloss:4f}  lr:{optimizer.param_groups[0]["lr"]:.5f}')
                if thresh == 0:
                    logger.log(f'{"name":<8}{"nums":>8}{"top1":>10}{"top5":>10}')
                    # val
                    top1, top5, v_loss = valuate(ema.ema, val_dataloader, device, pbar, True, criterion, logger, thresh)
                    logger.log(f'v_loss:{v_loss:4f}  mtop1:{top1:.3g}  mtop5:{top5:.3g}\n')
                else:
                    logger.log(f'{"name":<8}{"nums":>8}{"precision":>15}{"recall":>10}{"f1score":>10}')
                    # val
                    precision, recall, f1score, v_loss = valuate(ema.ema, val_dataloader, device, pbar, True, criterion, logger, thresh)
                    logger.log(f'v_loss:{v_loss:4f}  precision:{precision:.3g}  recall:{recall:.3g}  f1score:{f1score:.3g}\n')

                fitness = top1 if thresh == 0 else f1score  # define fitness as top1 accuracy

    schduler.step()  # step epoch-wise

    return fitness

