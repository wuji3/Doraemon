import torch
from tqdm import tqdm
from torch import Tensor
from engine.procedure.evaluation import valuate
from typing import Callable
import math
from engine.optimizer import SAM
from torch.utils.tensorboard import SummaryWriter

__all__ = ['Trainer']

def make_divisible(x: int, divisor = 32):
    # Returns nearest x divisible by divisor
    return math.ceil(x / divisor) * divisor

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

class Trainer:
    def __init__(self,
                 model,
                 train_dataloader,
                 val_dataloader,
                 optimizer,
                 scaler,
                 device: torch.device,
                 epochs: int,
                 logger,
                 rank: int,
                 scheduler,
                 ema, sampler = None,
                 thresh = 0,
                 teacher = None,
                 # face
                 print_freq = 50,
                 out_dir = None
                 ):

        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scaler = scaler
        self.device = device
        self.epochs = epochs
        self.logger = logger
        self.rank = rank
        self.scheduler = scheduler
        self.ema = ema
        self.sampler = sampler
        self.thresh = thresh
        self.teacher = teacher
        self.sam: bool = type(self.optimizer) is SAM
        self.distill: bool = teacher is not None

        # face
        self.print_freq = print_freq
        self.writer = SummaryWriter(log_dir=out_dir)

    def train_one_epoch(self, epoch: int, lam: float, criterion: Callable):
        # train mode
        self.model.train()

        cuda: bool = self.device != torch.device('cpu')

        if self.rank != -1:
            self.train_dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(self.train_dataloader)
        if self.rank in {-1, 0}:
            pbar = tqdm(enumerate(self.train_dataloader),
                        total=len(self.train_dataloader),
                        bar_format='{l_bar}{bar:10}{r_bar}')

        tloss, fitness = 0., 0.

        for i, (images, labels) in pbar:  # progress bar
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device)
            if self.sampler is not None:  # OHEM-Softmax
                with torch.no_grad():
                    valid = self.sampler.sample(self.model(images), labels)
                    images, labels = images[valid], labels[valid]
            with torch.cuda.amp.autocast(enabled=cuda):
                loss = self.compute_loss(images, labels, lam, criterion)

            if self.rank in {-1, 0}:
                tloss = (tloss * i + loss.item()) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if cuda else 0)  # (GB)
                pbar.desc = f"{f'{epoch + 1}/{self.epochs}':>10}{mem:>10}{tloss:>12.3g}" + ' ' * 36
                pbar.postfix = f'lr:{self.optimizer.param_groups[0]["lr"]:.5f}, imgsz:{print_imgsz(images)}'

                if i == len(pbar) - 1:  # last batch
                    self.logger.log(f'epoch:{epoch + 1:d}  t_loss:{tloss:4f}  lr:{self.optimizer.param_groups[0]["lr"]:.5f}')
                    if self.thresh == 0:
                        self.logger.log(f'{"name":<8}{"nums":>8}{"top1":>10}{"top5":>10}')
                        # val
                        top1, top5, v_loss = valuate(self.ema.ema, self.val_dataloader, self.device, pbar, True, criterion, self.logger,
                                                     self.thresh)
                        self.logger.log(f'v_loss:{v_loss:4f}  mtop1:{top1:.3g}  mtop5:{top5:.3g}\n')
                    else:
                        self.logger.log(f'{"name":<8}{"nums":>8}{"precision":>15}{"recall":>10}{"f1score":>10}')
                        # val
                        precision, recall, f1score, v_loss = valuate(self.ema.ema, self.val_dataloader, self.device, pbar, True,
                                                                     criterion, self.logger, self.thresh)
                        self.logger.log(
                            f'v_loss:{v_loss:4f}  precision:{precision:.3g}  recall:{recall:.3g}  f1score:{f1score:.3g}\n')

                    fitness = top1 if self.thresh == 0 else f1score  # define fitness as top1 accuracy

        self.scheduler.step()  # step epoch-wise

        return fitness

    @staticmethod
    def update_sam(model: torch.nn.Module, inputs, targets, optimizer, lossfn, rank, ema=None, mixup=False, **kwargs):
        # first forward-backward step
        optimizer.enable_running_stats(model)
        if not mixup:
            loss = lossfn(model(inputs), targets)
        else:
            loss = mixup_criterion(lossfn, model(inputs), **kwargs)
        if rank >= 0:  # multi-gpu
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

    def compute_loss(self, images: Tensor, labels: Tensor, lam: float, criterion: Callable, face: bool = False):
        mixup: bool = lam > 0

        assert not (mixup and self.distill), 'distill not be True when mixup is True'
        if mixup and self.sam: # close
            images, targets_a, targets_b = mixup_data(images, labels, self.device, lam)
            kwargs = dict(y_a=targets_a, y_b=targets_b, lam=lam)
            loss = Trainer.update_sam(self.model, images, labels, self.optimizer, criterion, self.rank, self.ema, mixup=True, **kwargs)
        elif mixup: # close
            images, targets_a, targets_b = mixup_data(images, labels, self.device, lam)
            loss = mixup_criterion(criterion, self.model(images), targets_a, targets_b, lam)
            Trainer.update(self.model, loss, self.scaler, self.optimizer, self.ema)
        elif self.sam and self.distill:
            pass
        elif self.sam: # close
            loss = Trainer.update_sam(self.model, images, labels, self.optimizer, criterion, self.rank, self.ema, mixup=False)
        elif self.distill:
            pass
        else: # close
            loss = criterion(self.model(images), labels) if not face else criterion(self.model(images, labels), labels)
            Trainer.update(self.model, loss, self.scaler, self.optimizer, self.ema)

        return loss

    # scale + backward + grad_clip + step + zero_gra
    @staticmethod
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

    def train_one_epoch_face(self, criterion, cur_epoch, loss_meter):
        """Tain one epoch by traditional training.
        """

        import os
        from copy import deepcopy

        iters_per_epoch = len(self.train_dataloader)

        for batch_idx, (images, labels) in enumerate(self.train_dataloader):
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device)

            loss = self.compute_loss(images, labels, lam=0, criterion = criterion, face = True)

            global_batch_idx = cur_epoch * iters_per_epoch + batch_idx
            self.scheduler.step() # step batch-wise

            torch.cuda.synchronize()
            loss_meter.update(loss.item(), images.shape[0])

            if self.rank in (-1, 0) and batch_idx % self.print_freq == 0:
                loss_avg = loss_meter.avg
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.both('Epoch %d, iter %d/%d, lr %f, loss %f' %
                            (cur_epoch, batch_idx, iters_per_epoch, lr, loss_avg))
                self.writer.add_scalar('Train_loss', loss_avg, global_batch_idx)
                self.writer.add_scalar('Train_lr', lr, global_batch_idx)
                loss_meter.reset()

            torch.cuda.empty_cache()

        if self.rank in (-1, 0):

            saved_name = 'Epoch_%d.pt' % cur_epoch
            ckpt = {
                'epoch': cur_epoch,
                'batch_id': batch_idx,
                # 'best_fitness': best_fitness,
                'model': self.model.state_dict() if self.rank == -1 else self.model.module.state_dict(),
                'ema': deepcopy(self.ema.ema),
                'updates': self.ema.updates,
                'optimizer': self.optimizer.state_dict(),  # optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            }
            torch.save(ckpt, os.path.join(self.writer.log_dir, saved_name))
            self.logger.both('Save checkpoint %s to disk...' % saved_name)

