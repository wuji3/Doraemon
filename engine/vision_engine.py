import torch
from datasets import load_dataset
from torchvision.transforms import CenterCrop, Resize, Compose, RandomChoice
from dataset.transforms import ResizeAndPadding2Square, RandomResizedCrop
import torch.nn as nn
from torch.cuda.amp import GradScaler
from dataset.basedataset import ImageDatasets
from dataset.transforms import CenterCropAndResize, SPATIAL_TRANSFORMS, create_AugTransforms
from torch.utils.data import DistributedSampler
from utils.logger import SmartLogger
from engine.optimizer import create_Optimizer
from engine.scheduler import create_Scheduler
from models.losses.loss import create_Lossfn
from engine.procedure.train import Trainer
from functools import reduce, partial
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from utils.plots import colorstr
from structure.sampler import OHEMImageSampler
from built.layer_optimizer import SeperateLayerParams
import time
import os
import datetime
import yaml
from copy import deepcopy
from models.ema import ModelEMA
from models import get_model, PreTrainedModels
from dataset.dataprocessor import SmartDataProcessor
from utils.average_meter import AverageMeter

__all__ = ['yaml_load', 'CenterProcessor','increment_path', 'check_cfgs_classification']


def yaml_load(file='data.yaml'):
    # Single-line safe yaml loading
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path

def check_cfgs_common(cfgs):
    def find_normalize(augment_list):
        for augment in augment_list:
            if 'normalize' in augment:
                return augment['normalize']
        return None

    hyp_cfg = cfgs['hyp']
    data_cfg = cfgs['data']
    model_cfg = cfgs['model']

    # image size
    # assert get_imgsz(cfgs['data']['train']['augment']) == get_imgsz(cfgs['data']['val']['augment']), 'imgsz should be same in training and inference'
    # loss
    assert reduce(lambda x, y: int(x) + int(y[0]), list(hyp_cfg['loss'].values())) == 1, 'ce or bce'
    # optimizer
    assert hyp_cfg['optimizer'][0] in {'sgd', 'adam', 'sam'}, 'optimizer choose sgd adam sam'
    # scheduler
    assert hyp_cfg['scheduler'] in {'linear', 'cosine', 'linear_with_warm', 'cosine_with_warm'}, 'scheduler support linear cosine linear_with_warm and cosine_with_warm'
    assert hyp_cfg['warm_ep'] >= 0 and isinstance(hyp_cfg['warm_ep'], int) and hyp_cfg['warm_ep'] < hyp_cfg['epochs'], 'warm_ep not be negtive, and should smaller than epochs'
    if hyp_cfg['warm_ep'] == 0: assert hyp_cfg['scheduler'] in {'linear', 'cosine'}, 'no warm, linear or cosine supported'
    if hyp_cfg['warm_ep'] > 0: assert hyp_cfg['scheduler'] in {'linear_with_warm', 'cosine_with_warm'}, 'with warm, linear_with_warm or cosine_with_warm supported'

    # normalize
    train_normalize = find_normalize(data_cfg['train']['augment'])
    val_normalize = find_normalize(data_cfg['val']['augment'])

    # Conditions for pretrained and non-pretrained models
    if model_cfg.get('pretrained', False) or model_cfg.get('load_from', 'False').startswith('torchvision'):
        # If pretrained, normalize must exist in both train and val, and their mean and std should be the same
        assert train_normalize is not None and val_normalize is not None, \
            'If pretrained, normalize must be present in both train and val'
        assert train_normalize['mean'] == val_normalize['mean'] and train_normalize['std'] == val_normalize['std'], \
            'Train and Val normalization parameters (mean, std) must be the same'
    elif 'load_from' in model_cfg and os.path.isfile(model_cfg['load_from']):
        pass
    else:
        # If not pretrained, normalize must not exist in either train or val
        assert train_normalize is None and val_normalize is None, \
            'If not pretrained, normalize should not be present in train or val'

def check_cfgs_face(cfgs):
    check_cfgs_common(cfgs=cfgs)

    model_cfg = cfgs['model']
    data_cfg = cfgs['data']
    # num_classes
    train_classes = [x for x in os.listdir(Path(data_cfg['root'])/'train') if not (x.startswith('.') or x.startswith('_'))]
    nc = model_cfg['head'][next(iter(model_cfg['head'].keys()))]['num_class']
    assert nc == len(train_classes), 'num_classes in model should be equal to classes in data folder'
    # pair_txt
    if cfgs['model']['task'] == 'face':
        assert os.path.isfile(data_cfg['val']['pair_txt']), 'make sure pair_txt exists'
        from engine.faceX.evaluation import Evaluator
        with open(data_cfg['val']['pair_txt']) as f:
            pair_list = [line.strip() for line in f.readlines()]
        Evaluator.check_nps(pair_list)
    # if cfgs['model']['task'] == 'cbir':
    #     for backbone in cfgs['model']['backbone']:
    #         image_size_backbone = int(cfgs['model']['backbone'][backbone]['image_size'])
    #         train_image_size = get_imgsz(cfgs['data']['train']['augment'])[0]
    #         assert image_size_backbone == train_image_size, f'image_size {image_size_backbone} in backbone should be equal to image_size {train_image_size} in data augment'
        
def check_cfgs_classification(cfgs):
    check_cfgs_common(cfgs=cfgs)

    model_cfg = cfgs['model']
    data_cfg = cfgs['data']
    hyp_cfg = cfgs['hyp']

    # Check num_classes
    if os.path.isdir(data_cfg['root']):
        # Local dataset
        train_classes = [x for x in os.listdir(Path(data_cfg['root'])/'train') if not (x.startswith('.') or x.startswith('_'))]
        num_classes = len(train_classes)
    else:
        # Assume it's a Hugging Face dataset
        try:
            dataset = load_dataset(data_cfg['root'], split='train')
            num_classes = len(set(dataset['label']))
        except Exception as e:
            raise ValueError(f"Failed to load dataset from {data_cfg['root']}. Error: {str(e)}")

    assert model_cfg['num_classes'] == num_classes, f'num_classes in model ({model_cfg["num_classes"]}) should be equal to classes in dataset ({num_classes})'

    # Check model
    assert model_cfg['name'].split('-')[0] in {'torchvision', 'custom'}, 'if from torchvision, torchvision-ModelName; if from your own, custom-ModelName'
    if model_cfg['kwargs'] and model_cfg['pretrained']:
        for k in model_cfg['kwargs'].keys():
            if k not in {'dropout','attention_dropout', 'stochastic_depth_prob'}:
                raise KeyError('set kwargs except dropout, pretrained must be False')

    # Check strategy
    # focalloss
    if hyp_cfg['strategy']['focal'][0]:
        assert hyp_cfg['loss']['bce'], 'focalloss only support bceloss'
    # ohem
    if hyp_cfg['strategy']['ohem'][0]:
        assert not hyp_cfg['loss']['bce'][0], 'ohem not support bceloss'
    # mixup
    mix_ratio, mix_duration = hyp_cfg['strategy']['mixup']["ratio"], hyp_cfg['strategy']['mixup']["duration"]
    assert 0 <= mix_ratio <= 1 and 0 <= mix_duration <= hyp_cfg['epochs'], f'mix_ratio should be in [0,1], mix_duration should be in [0, {hyp_cfg["epochs"]}]'
    hyp_cfg['strategy']['mixup'] = [mix_ratio, mix_duration]

def get_imgsz(augment: dict):
    augments = create_AugTransforms(augment)
    for a in augments.transforms[::-1]:
        if type(a) in SPATIAL_TRANSFORMS and hasattr(a, 'size'):
            if type(a.size) is int: return (a.size, a.size)
            elif type(a.size) in [tuple, list]: return tuple(a.size)
            else: raise ValueError('size be int, tuple or list')

class CenterProcessor:
    def __init__(self, cfgs: dict, rank: int, project: str = None, train: bool = True, opt = None):
        log_filename = Path(project) / "log{}.log".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) if project is not None else None
        self.project = project
        if rank in {-1, 0} and train:
            project.mkdir(parents=True, exist_ok=True)

        self.cfgs = cfgs
        self.model_cfg = cfgs['model']
        self.data_cfg = cfgs['data']
        self.hyp_cfg = cfgs['hyp']
        self.opt = opt
        self.imgsz = (cfgs['model']['image_size'], )

        # task
        self.task = self.model_cfg['task'] 

        # rank
        self.rank: int = rank
        # device
        if rank != -1:
            device = torch.device('cuda', rank)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device: torch.device = device

        # logger
        self.logger = SmartLogger(filename=log_filename, level=1) if rank in {-1,0} else None
        if self.logger is not None and rank in {-1, 0} and train:
            self.logger.console(cfgs) # output configs

        # model processor
        self.model_processor = get_model(self.model_cfg, self.logger, rank)
        self.model_processor.model.to(device)
        # data processor
        self.data_processor = SmartDataProcessor(self.data_cfg, rank=rank, project=project, training = train)
        if self.task == 'classification' and train:
            self.data_processor.val_dataset = self.data_processor.create_dataset('val', training = train)

        # loss
        loss_choice: str = 'ce' if self.hyp_cfg['loss']['ce'] else 'bce'
        self.loss_choice = loss_choice
        if self.task == 'classification':
            if train:
                self.lossfn = create_Lossfn(loss_choice)() \
                    if loss_choice == 'bce' \
                    else create_Lossfn(loss_choice)(label_smooth = self.hyp_cfg['label_smooth'])
            self.thresh = self.hyp_cfg['loss']['bce'][1] if loss_choice == 'bce' else 0

            # add label_transforms
            if loss_choice == 'bce':
                self.data_processor.train_dataset.label_transforms = \
                    partial(ImageDatasets.set_label_transforms,
                            num_classes = self.model_cfg['num_classes'],
                            label_smooth = self.hyp_cfg['label_smooth'])
                self.data_processor.val_dataset.label_transforms = \
                    partial(ImageDatasets.set_label_transforms,
                            num_classes=self.model_cfg['num_classes'],
                            label_smooth=0)
                # bce not support self.sampler
                self.sampler = None
                self.data_processor.train_dataset.multi_label = self.hyp_cfg['loss']['bce'][2]
                self.data_processor.val_dataset.multi_label = self.hyp_cfg['loss']['bce'][2]
            # ohem
            elif self.hyp_cfg['strategy']['ohem'][0]: self.sampler = OHEMImageSampler(*self.hyp_cfg['strategy']['ohem'][1:])
            else: self.sampler = None

        else: self.lossfn = create_Lossfn(loss_choice)(label_smooth = self.hyp_cfg['label_smooth'])

        if train and self.task == 'classification':
            # mixup sampler
            mixup_ratio, mixup_duration = self.hyp_cfg['strategy']['mixup']
            self.mixup_duration = mixup_duration
            self.mixup_ratio = mixup_ratio

            # progressive learning
            self.prog_learn = self.hyp_cfg['strategy']['prog_learn']

            if self.prog_learn: 
                warmup_epochs = self.hyp_cfg['warm_ep']
                remaining_epochs = self.hyp_cfg['epochs'] - warmup_epochs
                stage1_epochs = remaining_epochs // 4
                stage2_epochs = remaining_epochs // 4
                resize_chnodes = [
                    warmup_epochs, 
                    warmup_epochs + stage1_epochs,  
                    warmup_epochs + stage1_epochs + stage2_epochs
                ]
                self.resize_chnodes = resize_chnodes

                min_imgsz = min(self.imgsz)
                self.imgsz_milestone = torch.linspace(int(min_imgsz * 0.5), int(min_imgsz), 3, dtype=torch.int32).tolist()

            # focalloss hard
            if loss_choice == 'bce' and self.hyp_cfg['strategy']['focal'][0]:
                self.focal = create_Lossfn('focal')(gamma=self.hyp_cfg['strategy']['focal'][2], alpha= self.hyp_cfg['strategy']['focal'][1])
            else:
                self.focal = None

        # ema
        if train: self.ema = ModelEMA(self.model_processor.model) if rank in {-1, 0} else None

        self.loss_meter = AverageMeter()

    def set_optimizer_momentum(self, momentum) -> None:
        for g in self.optimizer.param_groups:
            g['momentum'] = momentum

    def auto_mixup(self, mixup: float, epoch:int, milestone: list) -> float:
        if mixup == 0 or epoch < milestone[0] or self.dist_sampler['beta'] is None : return 0
        else:
            mix_prob = self.dist_sampler['uniform'].sample()
            lam = self.dist_sampler['beta'].sample().to(self.device) if mix_prob < mixup else 0

            return lam

    def auto_prog(self, epoch: int):
        def create_AugSequence(train_augs : list, size):
            sequence = []
            for i, m in enumerate(train_augs):
                if isinstance(m, RandomChoice):
                    m.transforms = create_AugSequence(m.transforms, size)
                    sequence.append(m)
                elif isinstance(m, ResizeAndPadding2Square):
                    m.size = size
                    sequence.append(m)
                elif isinstance(m, CenterCrop):
                    if i + 1 < len(train_augs) and (not isinstance(train_augs[i + 1], Resize) or not isinstance(train_augs[i + 1], RandomResizedCrop)):
                        sequence.extend([m, Resize(size)])
                    else:
                        sequence.append(m)
                elif isinstance(m, Resize):
                    sequence.append(Resize(size))
                elif isinstance(m, CenterCropAndResize):
                    m[-1] = Resize(size)
                    sequence.append(m)
                elif isinstance(m, RandomResizedCrop):
                    m.size = (size, size)
                    m.resize_and_padding.size = size
                    sequence.append(m)
                else:
                    sequence.append(m)

            return sequence

        chnodes = self.resize_chnodes

        if epoch == chnodes[0]: size = self.imgsz_milestone[0]
        elif epoch == chnodes[1]: size = self.imgsz_milestone[1]
        elif epoch == chnodes[2]: size = self.imgsz_milestone[2]
        else: return

        if hasattr(self.data_processor.train_dataset.transforms, 'base_transforms'):
            transforms = self.data_processor.train_dataset.transforms.base_transforms.transforms
            self.data_processor.train_dataset.transforms.base_transforms = Compose(create_AugSequence(transforms, size))
        if hasattr(self.data_processor.train_dataset.transforms, 'class_transforms') and self.data_processor.train_dataset.transforms.class_transforms is not None:
            for c, transforms in self.data_processor.train_dataset.transforms.class_transforms.items():
                self.data_processor.train_dataset.transforms.class_transforms[c] = Compose(create_AugSequence(transforms.transforms, size))

    def set_sync_bn(self):
        self.model_processor.model = nn.SyncBatchNorm.convert_sync_batchnorm(module=self.model_processor.model)

    def run_classifier(self, resume = None): # train+val per epoch
        last, best = self.project / 'last.pt', self.project / 'best.pt'

        model, data_processor, scaler, device, epochs, logger, mixup_duration, rank, warm_ep, aug_epoch, focal, sampler, thresh = \
            self.model_processor.model, self.data_processor, \
            GradScaler(enabled = (self.device != torch.device('cpu'))), self.device, self.hyp_cfg['epochs'], \
            self.logger, self.mixup_duration, self.rank, self.hyp_cfg['warm_ep'], \
            self.data_cfg['train']['aug_epoch'], self.focal, self.sampler, self.thresh

        # data
        train_dataset, val_dataset = data_processor.train_dataset, data_processor.val_dataset
        data_sampler = None if self.rank == -1 else DistributedSampler(dataset=train_dataset)
        train_dataloader = data_processor.set_dataloader(dataset=train_dataset,
                                                         bs=self.data_cfg['train']['bs'],
                                                         nw=self.data_cfg['nw'],
                                                         pin_memory=True,
                                                         sampler=data_sampler,
                                                         shuffle=data_sampler is None,
                                                         collate_fn=train_dataset.collate_fn,
                                                         drop_last = True)

        if self.rank in {-1, 0}:
            val_dataloader = data_processor.set_dataloader(dataset=val_dataset,
                                                           bs=self.data_cfg['val']['bs'],
                                                           nw=self.data_cfg['nw'],
                                                           pin_memory=False,
                                                           shuffle=False,
                                                           collate_fn=val_dataset.collate_fn)
        else:
            val_dataloader = None

        # tell data distribution
        if rank in (-1, 0):
            ImageDatasets.tell_data_distribution({"train": train_dataset, "val": val_dataset}, logger, self.model_cfg['num_classes'], train_dataset.is_local_dataset)

        # optimizer
        params = SeperateLayerParams(model)
        optimizer = create_Optimizer(optimizer=self.hyp_cfg['optimizer'][0],
                                     lr=self.hyp_cfg['lr0'],
                                     weight_decay=self.hyp_cfg['weight_decay'],
                                     momentum=self.hyp_cfg['warmup_momentum'],
                                     params=params.create_ParamSequence(layer_wise=self.hyp_cfg['optimizer'][1],
                                                                        lr=self.hyp_cfg['lr0']))
        self.optimizer = optimizer
        # scheduler
        scheduler = create_Scheduler(scheduler=self.hyp_cfg['scheduler'],
                                     optimizer=optimizer,
                                     warm_ep=self.hyp_cfg['warm_ep'],
                                     epochs=self.hyp_cfg['epochs'],
                                     lr0=self.hyp_cfg['lr0'],
                                     lrf_ratio=self.hyp_cfg['lrf_ratio'])

        best_fitness = 0.
        start_epoch = 0

        # resume
        if resume is not None:
            ckp = torch.load(resume, map_location=device)
            start_epoch = ckp['epoch'] + 1
            best_fitness = ckp['best_fitness']
            if self.rank in {-1, 0}:
                self.ema.ema.load_state_dict(ckp['ema'].float().state_dict())
                self.ema.updates = ckp['updates']
            model.load_state_dict(ckp['model'])
            optimizer.load_state_dict(ckp['optimizer'])
            scheduler.load_state_dict(ckp['scheduler'])
            if device != torch.device('cpu'):
                scaler.load_state_dict(ckp['scaler'])

            if rank in (-1, 0): logger.both(f'resume: {resume}')

        if 'load_from' in self.model_cfg:
            load_from = self.model_cfg['load_from']
            state_dict = torch.load(load_from, weights_only=False)
            if 'ema' in state_dict: state_dict = state_dict['ema'].state_dict()
            else: 
                state_dict = state_dict['model_state_dict'].state_dict()
            missing_keys, unexpected_keys = model.load_state_dict(state_dict=state_dict, strict=False)
            if rank in (-1, 0): 
                logger.both(f'load_from: {load_from}')
                logger.both(f"Missing keys: {missing_keys}")
                logger.both(f"Unexpected keys: {unexpected_keys}")

        if rank != -1:
            model = DDP(model, device_ids=[self.rank])

        if self.rank in {-1, 0}:

            if thresh == 0:
                print(f"{'Epoch':>10}{'GPU_mem':>10}{'train_loss':>12}{f'val_loss':>12}{'top1_acc':>12}{'top5_acc':>12}")
            else:
                print(f"{'Epoch':>10}{'GPU_mem':>10}{'train_loss':>12}{f'val_loss':>12}{'precision':>12}{'recall':>12}{'f1score':>12}")
            time.sleep(0.2)

        # total epochs
        total_epoch = epochs

        # trainer
        trainer = Trainer(model, train_dataloader, val_dataloader, optimizer,
                          scaler, device, total_epoch, logger, rank, scheduler, self.ema, sampler, thresh,
                          self.teacher if hasattr(self, 'teacher') else None, 
                          None,
                          cfgs=self.cfgs)

        t0 = time.time()
        for epoch in range(start_epoch, total_epoch):
            # warmup set augment as val
            if epoch == 0:
                self.data_processor.set_augment('train', transforms=None)
                trainer.mixup_sampler = None

            # change optimizer momentum from warm_moment0.8 -> momentum0.937
            if epoch == warm_ep:
                self.set_optimizer_momentum(self.hyp_cfg['momentum'])
                self.data_processor.set_augment('train', transforms=create_AugTransforms(self.data_cfg['train']['augment']))
                if self.mixup_ratio == 0 or self.mixup_duration == 0:
                    trainer.mixup_sampler = None
                else:
                    trainer.mixup_sampler = torch.distributions.beta.Beta(self.mixup_ratio, self.mixup_ratio)
                    if rank in {-1, 0}:
                        logger.both(f"Mixup start up")
            
            if (self.mixup_ratio != 0 and self.mixup_duration != 0) and epoch == warm_ep + mixup_duration:
                trainer.mixup_sampler = None
                if rank in {-1, 0}:
                    logger.both(f"Mixup end")

            # change lossfn bce -> focal
            if epoch == warm_ep and self.focal is not None:
                self.lossfn = self.focal
            
            # weaken data augment at milestone
            self.data_processor.auto_aug_weaken(int(epoch), milestone=aug_epoch)

            # progressive learning: effect on imagesz & mixup
            if self.prog_learn:
                self.auto_prog(epoch=epoch)

            # train for one epoch
            fitness = trainer.train_one_epoch(epoch, self.lossfn)

            if rank in {-1, 0}:
                # Best fitness
                if fitness > best_fitness:
                    best_fitness = fitness

                # Save model
                final_epoch: bool = epoch + 1 == total_epoch
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': model.state_dict() if rank == -1 else model.module.state_dict(),  # deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(self.ema.ema),
                    'updates': self.ema.updates,
                    'optimizer': optimizer.state_dict(),  # optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                if device != torch.device('cpu'):
                    ckpt['scaler'] = scaler.state_dict()

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fitness:
                    torch.save(ckpt, best)
                del ckpt

                # complete
                if final_epoch:
                    logger.both(f'\nTraining complete ({(time.time() - t0) / 3600:.3f} hours)'
                                   f"\nResults saved to {colorstr('bold', self.project)}"
                                   f'\nPredict:         python visualize.py --cfgs {os.path.join(os.path.dirname(best), os.path.basename(self.opt.cfgs))} --weight {best} --badcase --class_json {self.project}/class_indices.json --ema --cam --data {data_processor.data_cfgs["root"]}/val/{colorstr("blue", "XXX_cls")}'
                                   f'\nValidate:        python validate.py --cfgs {os.path.join(os.path.dirname(best), os.path.basename(self.opt.cfgs))} --eval_topk 5 --weight {best} --ema' )

    def run_face(self, resume = None):
        model, data_processor, scaler, device, epochs, logger, rank, warm_ep, aug_epoch, task = self.model_processor.model, self.data_processor, \
            GradScaler(enabled = (self.device != torch.device('cpu'))), self.device, self.hyp_cfg['epochs'], self.logger, self.rank, self.hyp_cfg['warm_ep'], \
            self.data_cfg['train']['aug_epoch'], self.model_cfg['task']

        # load for fine-tune
        if 'load_from' in self.model_cfg:
            load_from = self.model_cfg['load_from']
            if load_from.startswith("torchvision") and load_from in PreTrainedModels:
                modelname = load_from.split('-')[1]
                from torchvision.models import get_model as torchvision_get_model
                if rank in (-1, 0): # make sure rank 0 download once
                    torchvision_get_model(modelname, weights = PreTrainedModels[load_from])
                if rank >= 0: 
                    dist.barrier()
                state_dict = torchvision_get_model(modelname, weights = PreTrainedModels[load_from]).state_dict()
            else:
                state_dict = torch.load(load_from, weights_only=False)
                if 'ema' in state_dict: state_dict = state_dict['ema']
                else: 
                    state_dict = state_dict['model_state_dict']
            missing_keys, unexpected_keys = model.trainingwrapper['backbone'].load_state_dict(state_dict=state_dict, strict=False)
            if rank in (-1, 0): 
                logger.both(f'load_from: {load_from}')
                logger.both(f"Missing keys: {missing_keys}")
                logger.both(f"Unexpected keys: {unexpected_keys}")

        # data
        train_dataset = data_processor.train_dataset
        data_sampler = None if self.rank == -1 else DistributedSampler(dataset=train_dataset)
        train_dataloader = data_processor.set_dataloader(dataset=train_dataset,
                                                         bs=self.data_cfg['train']['bs'],
                                                         nw=self.data_cfg['nw'],
                                                         pin_memory=True,
                                                         sampler=data_sampler,
                                                         shuffle=data_sampler is None,
                                                         collate_fn=train_dataset.collate_fn,
                                                         drop_last = True)
        # tell data distribution
        if self.rank in (-1, 0):
            ImageDatasets.tell_data_distribution(self.data_cfg['root'], logger, self.model_cfg['head'][next(iter(self.model_cfg['head'].keys()))]['num_class'])

        # optimizer
        params = SeperateLayerParams(model)
        optimizer = create_Optimizer(optimizer=self.hyp_cfg['optimizer'][0],
                                     lr=self.hyp_cfg['lr0'],
                                     weight_decay=self.hyp_cfg['weight_decay'],
                                     momentum=self.hyp_cfg['warmup_momentum'],
                                     params=params.create_ParamSequence(layer_wise=self.hyp_cfg['optimizer'][1],
                                                                        lr=self.hyp_cfg['lr0']))
        self.optimizer = optimizer
        # scheduler
        scheduler = create_Scheduler(scheduler=self.hyp_cfg['scheduler'],
                                     optimizer=optimizer,
                                     warm_ep=self.hyp_cfg['warm_ep'] * len(train_dataloader),
                                     epochs=self.hyp_cfg['epochs'] * len(train_dataloader),
                                     lr0=self.hyp_cfg['lr0'],
                                     lrf_ratio=self.hyp_cfg['lrf_ratio'])

        start_epoch = 0

        # resume
        if resume is not None:
            ckp = torch.load(resume, map_location=device)
            start_epoch = ckp['epoch'] + 1

            if self.rank in {-1, 0}:
                self.ema.ema.load_state_dict(ckp['ema'].float().state_dict())
                self.ema.updates = ckp['updates']
            model.load_state_dict(ckp['state_dict'])
            optimizer.load_state_dict(ckp['optimizer'])
            scheduler.load_state_dict(ckp['scheduler'])
            if device != torch.device('cpu'):
                scaler.load_state_dict(ckp['scaler'])

            if rank in (-1, 0): logger.both(f'resume: {resume}')

        if rank != -1:
            model = DDP(model, device_ids=[self.rank])

        if self.rank in {-1, 0}: time.sleep(0.2)

        # total epochs
        total_epoch = epochs

        # trainer
        trainer = Trainer(model=model, 
                          train_dataloader=train_dataloader, 
                          val_dataloader=None, 
                          optimizer=optimizer,
                          scaler=scaler, 
                          device=device, 
                          epochs=total_epoch, 
                          logger=logger, 
                          rank=rank, 
                          scheduler=scheduler, 
                          ema=self.ema, 
                          sampler=None, 
                          thresh=None,
                          teacher=self.teacher if hasattr(self, 'teacher') else None, 
                          task=task, 
                          print_freq=self.opt.print_freq, 
                          save_freq=self.opt.save_freq, 
                          cfgs=self.cfgs, 
                          out_dir=self.opt.save_dir)

        t0 = time.time()
        for epoch in range(start_epoch, total_epoch):
            # warmup set augment as val
            if epoch == 0:
                self.data_processor.set_augment('train', transforms=create_AugTransforms(self.data_cfg['val']['augment']))

            # change optimizer momentum from warm_moment0.8 -> momentum0.937
            if epoch == warm_ep:
                self.set_optimizer_momentum(self.hyp_cfg['momentum'])
                self.data_processor.set_augment('train', transforms=create_AugTransforms(self.data_cfg['train']['augment']))

            # weaken data augment at milestone
            self.data_processor.auto_aug_weaken(epoch, milestone=aug_epoch, sequence = create_AugTransforms(self.data_cfg['val']['augment']))

            # train for one epoch
            trainer.train_one_epoch_face(self.lossfn, epoch, self.loss_meter)

        if rank in (-1, 0):
            logger.both(f'\nTraining complete ({(time.time() - t0) / 3600:.3f} hours)'
                        f"\nResults saved to {colorstr('bold', self.project)}"
                        f'\nValidate:        python validate.py --cfgs {self.opt.cfgs} --weight {self.project}/{colorstr("blue", "which_weight")} --ema')
