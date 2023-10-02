from utils.general import SmartDataProcessor, SmartModel, SmartLogger, Datasets
import os
import argparse
from pathlib import Path
from utils.valuate import val
import torch
from functools import partial

RANK = int(os.getenv('RANK', -1))
ROOT = Path(os.path.dirname(__file__))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default = ROOT / 'data', help='data/val')
    parser.add_argument('--choice', default = 'torchvision-shufflenet_v2_x1_0', type=str)
    parser.add_argument('--thresh', default = 0.7, type=float)
    parser.add_argument('--head', default = 'ce', type=str)
    parser.add_argument('--multi_label', default = False, type=bool)
    parser.add_argument('--num_classes', default = 6, type=int)
    parser.add_argument('--kwargs', default = "{}", type=str, )
    parser.add_argument('--weight', default = './run/exp/best.pt', help='configs for models, data, hyps')
    parser.add_argument('--transforms', default = {'to_tensor': 'no_params', 'normalize': 'no_params'}, help='空格隔开')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    return parser.parse_args()

def main(opt):
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data
    data_cfgs = {}
    data_cfgs['root'] = opt.root
    data_cfgs['val'] = {'augment': opt.transforms if isinstance(opt.transforms, dict) else eval(opt.transforms)}

    data_processor = SmartDataProcessor(data_cfgs=data_cfgs, rank=RANK, project=None)
    data_processor.val_dataset = data_processor.create_dataset('val')
    if opt.head == 'bce':
        data_processor.val_dataset.multi_label = opt.multi_label
        data_processor.val_dataset.label_transforms = \
            partial(Datasets.set_label_transforms,
                    num_classes=opt.num_classes,
                    label_smooth=0)
    dataloader = data_processor.set_dataloader(data_processor.val_dataset, bs=8, collate_fn=data_processor.val_dataset.collate_fn) # batchsize default 256

    # model
    model_cfg = {}
    model_cfg['choice'] = opt.choice
    model_cfg['num_classes'] = opt.num_classes
    model_cfg['kwargs'] = eval(opt.kwargs)
    model_cfg['pretrained'] = True
    model_cfg['backbone_freeze'] = False
    model_cfg['bn_freeze'] = False
    model_cfg['bn_freeze_affine'] = False

    model_processor = SmartModel(model_cfg)
    model = model_processor.model
    weights = torch.load(opt.weight, map_location=device)['ema'].float().state_dict()
    model.load_state_dict(weights)
    model.to(device)

    # logger
    logger = SmartLogger()

    # val
    val(model, dataloader, device, None, False, None, logger, thresh=opt.thresh)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
