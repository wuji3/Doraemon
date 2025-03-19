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
from prettytable import PrettyTable


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
        assert len(thresh) == len(dataloader.dataset.id2label), \
            f'Number of thresholds ({len(thresh)}) must match number of classes ({len(dataloader.dataset.id2label)})'
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
            thresh = torch.full((len(dataloader.dataset.id2label),), 
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
                    hard_labels = (labels >= 0.5).float()
                    targets.append(hard_labels)
                if lossfn:
                    loss += lossfn(y, labels)

    loss /= n
    pred, targets = torch.cat(pred), torch.cat(targets)
    
    if not is_training and is_single_label and len(dataloader.dataset.id2label) <= 10:
        conm = ConfusedMatrix(len(dataloader.dataset.id2label))
        conm.update(targets, pred[:, 0])
        conm.save_conm(conm.mat.detach().cpu().numpy(), dataloader.dataset.id2label, conm_path if conm_path is not None else 'conm.png')

    if is_single_label:
        correct = (targets[:, None] == pred).float()
        acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) accuracy
        top1, top5 = acc.mean(0).tolist()
        
        if not is_training:
            table = PrettyTable(['Class', 'Samples', 'Top1', f'Top{top_k}'])
            
            for i, c in dataloader.dataset.id2label.items():
                acc_i = acc[targets == i]
                top1i, top5i = acc_i.mean(0).tolist()
                table.add_row([c, acc_i.shape[0], f'{top1i:.3f}', f'{top5i:.3f}'])
            
            table.add_row(['MEAN', acc.shape[0], f'{top1:.3f}', f'{top5:.3f}'])
            
            logger.console('\n' + str(table))
        else:
            table = PrettyTable(['Class', 'Samples', 'Top1', f'Top{top_k}'])
            for i, c in dataloader.dataset.id2label.items():
                acc_i = acc[targets == i]
                top1i, top5i = acc_i.mean(0).tolist()
                table.add_row([c, acc_i.shape[0], f'{top1i:.3f}', f'{top5i:.3f}'])
            table.add_row(['MEAN', acc.shape[0], f'{top1:.3f}', f'{top5:.3f}'])
            logger.log('\n' + str(table))
    else:
        num_classes = len(dataloader.dataset.id2label)
        # Compute precision, recall, and F1-score for each class
        precisioner = Precision(task='multilabel', threshold=0.5, num_labels=num_classes, average=None).to(device)
        recaller = Recall(task='multilabel', threshold=0.5, num_labels=num_classes, average=None).to(device)
        f1scorer = F1Score(task='multilabel', threshold=0.5, num_labels=num_classes, average=None).to(device)

        # Compute precision, recall, and F1-score for each class
        precision = precisioner(pred.float(), targets)
        recall = recaller(pred.float(), targets)
        f1score = f1scorer(pred.float(), targets)

        cls_numbers = targets.sum(0).int().tolist()
        
        if is_training:
            table = PrettyTable(['Class', 'Samples', 'Precision', 'Recall', 'F1-Score', 'Threshold'])
            for i, c in dataloader.dataset.id2label.items():
                table.add_row([
                    c, 
                    cls_numbers[i], 
                    f'{precision[i].item():.3f}',
                    f'{recall[i].item():.3f}',
                    f'{f1score[i].item():.3f}',
                    f'{thresh[i].item():.3f}' if isinstance(thresh, torch.Tensor) else f'{thresh:.3f}'
                ])
            table.add_row([
                'MEAN',
                sum(cls_numbers),
                f'{precision.mean().item():.3f}',
                f'{recall.mean().item():.3f}',
                f'{f1score.mean().item():.3f}',
                '-'
            ])
            logger.log('\n' + str(table))
        else:
            table = PrettyTable(['Class', 'Samples', 'Precision', 'Recall', 'F1-Score', 'Threshold'])
            
            for i, c in dataloader.dataset.id2label.items():
                table.add_row([
                    c, 
                    cls_numbers[i], 
                    f'{precision[i].item():.3f}',
                    f'{recall[i].item():.3f}',
                    f'{f1score[i].item():.3f}',
                    f'{thresh[i].item():.3f}' if isinstance(thresh, torch.Tensor) else f'{thresh:.3f}'
                ])
            
            table.add_row([
                'MEAN',
                sum(cls_numbers),
                f'{precision.mean().item():.3f}',
                f'{recall.mean().item():.3f}',
                f'{f1score.mean().item():.3f}',
                '-'
            ])
            
            # 显示表格
            logger.console('\n' + str(table))

    if pbar:
        if is_single_label:
            pbar.desc = f'{pbar.desc[:-36]}{loss:>12.3g}{top1:>12.3g}{top5:>12.3g}'
        else:
            pbar.desc = f'{pbar.desc[:-36]}{loss:>12.3g}{precision.mean().item():>12.3g}{recall.mean().item():>12.3g}{f1score.mean().item():>12.3g}'

    # filename = 'train_results.txt' if is_training else 'val_results.txt'
    # with open(filename, 'w') as f:
    #     if is_single_label:
    #         # Save each sample's prediction and true label
    #         for i in range(len(targets)):
    #             f.write(f'Sample {i}: GT={targets[i].item()}, Pred={pred[i, 0].item()}, Top{top_k}={[pred[i, j].item() for j in range(top_k)]}\n')
    #     else:
    #         # Save each sample's multi-label prediction and true label
    #         for i in range(len(targets)):
    #             gt = targets[i].cpu().numpy()
    #             pd = pred[i].cpu().numpy()
    #             gt_indices = np.where(gt == 1)[0]
    #             pred_indices = np.where(pd == 1)[0]
    #             f.write(f'Sample {i}:\n')
    #             f.write(f'  GT classes: {gt_indices.tolist()}\n')
    #             f.write(f'  Pred classes: {pred_indices.tolist()}\n')

    if lossfn:
        if is_single_label: 
            return top1, top5, loss
        else: 
            return precision.mean().item(), recall.mean().item(), f1score.mean().item(), loss
    else:
        if is_single_label: 
            return top1, top5
        else: 
            return precision.mean().item(), recall.mean().item(), f1score.mean().item()
