import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm
from typing import Callable, Optional
from torchmetrics import Precision, Recall, F1Score

__all__ = ['val']
def val(model: nn.Module, dataloader, device: torch.device, pbar, is_training: bool = False, lossfn: Optional[Callable] = None, logger = None, thresh: float = 0):

    assert thresh == 0 or thresh > 0 and thresh < 1, 'softmax时thresh为0 bce时0 < thresh < 1'
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
                if thresh == 0:
                    pred.append(y.argsort(1, descending=True)[:, :5])
                    targets.append(labels)
                else:
                    pred.append(y.sigmoid())
                    # turn label_smoothing to onehot 硬标签
                    hard_labels = labels.round()
                    hard_labels = torch.where(hard_labels == 1, hard_labels, 0)
                    targets.append(hard_labels)
                if lossfn:
                    loss += lossfn(y, labels)

    loss /= n
    pred, targets = torch.cat(pred), torch.cat(targets)
    # if targets.dim() > 1: targets = torch.argmax(targets, dim=-1) # bce label, only used in training in order to compute loss
    if thresh == 0:
        correct = (targets[:, None] == pred).float()
        acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) accuracy
        top1, top5 = acc.mean(0).tolist()
        for i, c in enumerate(dataloader.dataset.class_indices):
            acc_i = acc[targets == i]
            top1i, top5i = acc_i.mean(0).tolist()
            if not is_training: logger.console(f'{c:<8}{acc_i.shape[0]:>8}{top1i:>10.3f}{top5i:>10.3f}')
            else: logger.log(f'{c:<8}{acc_i.shape[0]:>8}{top1i:>10.3f}{top5i:>10.3f}')
    else:
        num_classes = len(dataloader.dataset.class_indices)
        precisioner = Precision(task='multilabel', threshold=thresh, num_labels=num_classes, average=None).to(device)
        recaller = Recall(task='multilabel', threshold=thresh, num_labels=num_classes, average=None).to(device)
        f1scorer = F1Score(task='multilabel', threshold=thresh, num_labels=num_classes, average=None).to(device)

        precision = precisioner(pred, targets)
        recall = recaller(pred, targets)
        f1score = f1scorer(pred, targets)

        if not is_training: logger.console(f'{"name":<8}{"nums":>8}{"precision":>10}{"recall":>10}{"f1-score":>10}')
        for i, c in enumerate(dataloader.dataset.class_indices):
            if not is_training: logger.console(f'{c:<8}{pred.shape[0]:>8}{precision[i].item():>10.3f}{recall[i].item():>10.3f}{f1score[i].item():>10.3f}')
            else: logger.log(f'{c:<8}{pred.shape[0]:>8}{precision[i].item():>15.3f}{recall[i].item():>10.3f}{f1score[i].item():>10.3f}')

    if pbar and thresh == 0:
        pbar.desc = f'{pbar.desc[:-36]}{loss:>12.3g}{top1:>12.3g}{top5:>12.3g}'
    elif pbar and thresh > 0:
        pbar.desc = f'{pbar.desc[:-36]}{loss:>12.3g}{precision.mean().item():>12.3g}{recall.mean().item():>12.3g}{f1score.mean().item():>12.3g}'

    if not is_training: logger.console(f'mprecision:{precision.mean().item():.3f}, mrecall:{recall.mean().item():.3f}, mf1-score:{f1score.mean().item():.3f},')

    if lossfn:
        if thresh == 0: return top1, top5, loss
        else: return precision.mean().item(), recall.mean().item(), f1score.mean().item(), loss
    else:
        if thresh == 0: return top1, top5
        else: return precision.mean().item(), recall.mean().item(), f1score.mean().item(),

