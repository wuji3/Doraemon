import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm
from typing import Callable, Optional


__all__ = ['val']
def val(model: nn.Module, dataloader, device: torch.device, pbar, is_training: bool = False, lossfn: Optional[Callable] = None, logger = None):

    # eval mode
    model.eval()

    n = len(dataloader)  # number of batches
    action = 'validating'
    desc = f'{pbar.desc[:-36]}{action:>36}' if pbar else f'{action}'
    bar = tqdm(dataloader, desc, n, not is_training, bar_format='{l_bar}{bar:10}{r_bar}', position=0)
    pred, targets, loss = [], [], 0
    with torch.no_grad(): # w/o this op, computation graph will be save
        with autocast(enabled=(device != torch.device('cpu'))):
            for images, labels in bar:
                images, labels = images.to(device, non_blocking = True), labels.to(device)
                y = model(images)
                pred.append(y.argsort(1, descending=True)[:, :5])
                targets.append(labels)
                if lossfn:
                    loss += lossfn(y, labels)

    loss /= n
    pred, targets = torch.cat(pred), torch.cat(targets)
    if targets.dim() > 1: targets = torch.argmax(targets, dim=-1) # bce label, only used in training in order to compute loss
    correct = (targets[:, None] == pred).float()
    acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) accuracy
    top1, top5 = acc.mean(0).tolist()

    if pbar:
        pbar.desc = f'{pbar.desc[:-36]}{loss:>12.3g}{top1:>12.3g}{top5:>12.3g}'

    if not is_training: logger.console(f'{"name":<8}{"nums":>8}{"top1":>10}{"top5":>10}')
    for i, c in enumerate(dataloader.dataset.class_indices):
        acc_i = acc[targets == i]
        top1i, top5i = acc_i.mean(0).tolist()
        if not is_training: logger.console(f'{c:<8}{acc_i.shape[0]:>8}{top1i:>10.3f}{top5i:>10.3f}')
        else: logger.log(f'{c:<8}{acc_i.shape[0]:>8}{top1i:>10.3f}{top5i:>10.3f}')

    if not is_training: logger.console(f'mtop1:{top1:.3f}, mtop5:{top5:.3f}')

    if lossfn: return top1, top5, loss
    else: return top1, top5

