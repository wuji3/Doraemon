from engine.vision_engine import SmartLogger, BaseDatasets
from dataset.dataprocessor import SmartDataProcessor
from models import SmartModel
import os
import argparse
from pathlib import Path
from engine.procedure.evaluation import valuate
import torch
from functools import partial

RANK = int(os.getenv('RANK', -1))
ROOT = Path(os.path.dirname(__file__))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default = ROOT / 'data', help='data/val')
    parser.add_argument('--choice', default = 'torchvision-shufflenet_v2_x1_0', type=str)
    parser.add_argument('--batchsize', default = 8, type=int)
    parser.add_argument('--nw', default=4, type=int, help='num_workers in dataloader')
    parser.add_argument('--thresh', default = 0.7, type=float)
    parser.add_argument('--head', default = 'ce', type=str)
    parser.add_argument('--eval_topk', default = 5, type=int)
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--multi_label', action='store_true')
    parser.add_argument('--num_classes', default = 6, type=int)
    parser.add_argument('--kwargs', default = "{}", type=str, )
    parser.add_argument('--weight', default = './run/exp/best.pt', help='configs for models, data, hyps')
    parser.add_argument('--transforms', default = {'to_tensor': 'no_params', 'normalize': 'no_params'}, help='空格隔开')
    parser.add_argument('--attention_pool', action='store_true', help='是否使用注意力池化, 默认False, 即使用平均池化')
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
            partial(BaseDatasets.set_label_transforms,
                    num_classes=opt.num_classes,
                    label_smooth=0)
    dataloader = data_processor.set_dataloader(data_processor.val_dataset, nw = opt.nw , bs=opt.batchsize, collate_fn=data_processor.val_dataset.collate_fn)

    # model
    model_cfg = {}
    model_cfg['choice'] = opt.choice
    model_cfg['num_classes'] = opt.num_classes
    model_cfg['kwargs'] = eval(opt.kwargs)
    model_cfg['pretrained'] = True
    model_cfg['backbone_freeze'] = False
    model_cfg['bn_freeze'] = False
    model_cfg['bn_freeze_affine'] = False
    model_cfg['attention_pool'] = opt.attention_pool

    model_processor = SmartModel(model_cfg)
    model = model_processor.model
    if opt.ema:
        weights = torch.load(opt.weight, map_location=device)['ema'].float().state_dict()
    else:
        weights = torch.load(opt.weight, map_location=device)['model']
    model.load_state_dict(weights)
    model.to(device)

    # logger
    logger = SmartLogger()

    # val
    valuate(model, dataloader, device, None, False, None, logger, thresh=opt.thresh, top_k=opt.eval_topk)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
