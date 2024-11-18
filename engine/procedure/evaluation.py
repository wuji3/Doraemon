import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm
from typing import Callable, Optional, Union, List
from torchmetrics import Precision, Recall, F1Score
from torch import Tensor
import itertools
import matplotlib.pyplot as plt
import numpy as np
from typing import Sequence


__all__ = ['valuate']

class ConfusedMatrix:
    def __init__(self, nc: int):
        self.nc = nc
        self.mat = None

    def update(self, gt: Tensor, pred: Tensor):
        if self.mat is None: self.mat = torch.zeros((self.nc, self.nc), dtype=torch.int64, device = gt.device)

        idx = gt * self.nc + pred
        self.mat += torch.bincount(idx, minlength=self.nc).reshape(self.nc, self.nc)

    def save_conm(self, cm: np.ndarray, classes: Sequence, save_path: str, cmap=plt.cm.cool):
        """
        - cm : 计算出的混淆矩阵的值
        - classes : 混淆矩阵中每一行每一列对应的列
        - normalize : True:显示百分比, False:显示个数
        """
        ax = plt.gca()
        ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.colorbar()
        tick_marks = [x for x in range(len(classes))]
        plt.xticks(tick_marks, classes, rotation=0, fontsize=10)
        plt.yticks(tick_marks, classes, fontsize=10)
        fmt = '.2f'
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="black")
        plt.tight_layout()
        plt.ylabel('GT', fontsize=12)
        plt.xlabel('Predict', fontsize=12)
        ax.xaxis.set_label_position('top')
        plt.gcf().subplots_adjust(top=0.9)
        plt.savefig(save_path)

def valuate(model: nn.Module, dataloader, device: torch.device, pbar, 
            is_training: bool = False, lossfn: Optional[Callable] = None, 
            logger = None, thresh: Union[float, List[float]] = 0, 
            top_k: int = 5, conm_path: str = None):
    """
    Evaluate model performance
    
    Args:
        ...
        thresh: float or List[float], threshold for each class in multi-label classification
               - float: same threshold for all classes
               - List[float]: specific threshold for each class
        ...
    """
    is_single_label = isinstance(thresh, (int, float)) and thresh == 0

    # Check threshold type
    if isinstance(thresh, (list, tuple, np.ndarray)):
        # Multi-threshold case: each class uses a different threshold
        assert len(thresh) == len(dataloader.dataset.class_indices), \
            f'Number of thresholds ({len(thresh)}) must match number of classes ({len(dataloader.dataset.class_indices)})'
        thresh = torch.tensor(thresh, device=device)
        # Verify that all thresholds are within the valid range
        assert (thresh > 0).all() and (thresh < 1).all(), \
            'For multi-label (BCE), all thresholds should be in (0, 1)'
    elif isinstance(thresh, (int, float)):
        if is_single_label:
            # Single-label classification (Softmax) case
            pass
        else:
            # Multi-label classification (BCE), use the same threshold
            assert 0 < thresh < 1, 'For multi-label (BCE), threshold should be in (0, 1)'
            thresh = torch.full((len(dataloader.dataset.class_indices),), 
                              thresh, 
                              device=device)
    else:
        raise ValueError(f'Unsupported threshold type: {type(thresh)}. '
                        f'Expected float or list/tuple/ndarray of floats.')

    # eval mode
    model.eval()

    n = len(dataloader)  # number of batches
    action = 'validating'
    desc = f'{pbar.desc[:-36]}{action:>36}' if pbar else f'{action}'
    bar = tqdm(dataloader, desc, n, not is_training, bar_format='{l_bar}{bar:10}{r_bar}', position=0)
    pred, targets, loss = [], [], 0
    
    with torch.no_grad():
        with autocast('cuda', enabled=(device != torch.device('cpu'))):
            for images, labels in bar:
                images, labels = images.to(device, non_blocking=True), labels.to(device)
                y = model(images)
                if is_single_label:
                    pred.append(y.argsort(1, descending=True)[:, :top_k])
                    targets.append(labels)
                else:
                    # Get prediction probabilities using sigmoid
                    pred_prob = y.sigmoid()
                    # Predict using threshold for each class
                    pred.append(pred_prob >= thresh)
                    # Convert to hard labels
                    hard_labels = labels.round()
                    hard_labels = torch.where(hard_labels == 1, hard_labels, 0)
                    targets.append(hard_labels)
                if lossfn:
                    loss += lossfn(y, labels)

    loss /= n
    pred, targets = torch.cat(pred), torch.cat(targets)
    
    if not is_training and is_single_label and len(dataloader.dataset.class_indices) <= 10:
        conm = ConfusedMatrix(len(dataloader.dataset.class_indices))
        conm.update(targets, pred[:, 0])
        conm.save_conm(conm.mat.detach().cpu().numpy(), dataloader.dataset.class_indices, conm_path if conm_path is not None else 'conm.png')

    if is_single_label:
        correct = (targets[:, None] == pred).float()
        acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) accuracy
        top1, top5 = acc.mean(0).tolist()
        if not is_training: logger.console(f'{"name":<15}{"nums":>8}{"top1":>10}{f"top{top_k}":>10}')
        for i, c in enumerate(dataloader.dataset.class_indices):
            acc_i = acc[targets == i]
            top1i, top5i = acc_i.mean(0).tolist()
            if not is_training: logger.console(f'{c:<15}{acc_i.shape[0]:>8}{top1i:>10.3f}{top5i:>10.3f}')
            else: logger.log(f'{c:<8}{acc_i.shape[0]:>8}{top1i:>10.3f}{top5i:>10.3f}')
        if not is_training: logger.console(f'{"    ":<15}{acc.shape[0]:>8}{top1:>10.3f}{round(top5, 3):>10.3f}')
    else:
        num_classes = len(dataloader.dataset.class_indices)
        # Compute precision, recall, and F1-score for each class
        precisioner = Precision(task='multilabel', threshold=0.5, num_labels=num_classes, average=None).to(device)
        recaller = Recall(task='multilabel', threshold=0.5, num_labels=num_classes, average=None).to(device)
        f1scorer = F1Score(task='multilabel', threshold=0.5, num_labels=num_classes, average=None).to(device)

        # Compute precision, recall, and F1-score for each class
        precision = precisioner(pred.float(), targets)
        recall = recaller(pred.float(), targets)
        f1score = f1scorer(pred.float(), targets)

        if not is_training: 
            logger.console(f'{"name":<8}{"nums":>8}{"precision":>10}{"recall":>10}{"f1-score":>10}{"thresh":>10}')
        cls_numbers = targets.sum(0).int().tolist()
        for i, c in enumerate(dataloader.dataset.class_indices):
            if not is_training: 
                logger.console(
                    f'{c:<8}{cls_numbers[i]:>8}{precision[i].item():>10.3f}'
                    f'{recall[i].item():>10.3f}{f1score[i].item():>10.3f}'
                    f'{thresh[i].item():>10.3f}'
                )
            else: 
                logger.log(
                    f'{c:<8}{cls_numbers[i]:>8}{precision[i].item():>15.3f}'
                    f'{recall[i].item():>10.3f}{f1score[i].item():>10.3f}'
                )

        if not is_training: 
            logger.console(
                f'mprecision:{precision.mean().item():.3f}, '
                f'mrecall:{recall.mean().item():.3f}, '
                f'mf1-score:{f1score.mean().item():.3f}'
            )

    if pbar:
        if is_single_label:
            pbar.desc = f'{pbar.desc[:-36]}{loss:>12.3g}{top1:>12.3g}{top5:>12.3g}'
        else:
            pbar.desc = f'{pbar.desc[:-36]}{loss:>12.3g}{precision.mean().item():>12.3g}{recall.mean().item():>12.3g}{f1score.mean().item():>12.3g}'

    if lossfn:
        if is_single_label: return top1, top5, loss
        else: return precision.mean().item(), recall.mean().item(), f1score.mean().item(), loss
    else:
        if is_single_label: return top1, top5
        else: return precision.mean().item(), recall.mean().item(), f1score.mean().item()
