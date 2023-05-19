from utils.general import SmartDataProcessor, SmartModel, SmartLogger
import os
import argparse
from pathlib import Path
from utils.valuate import val
import torch

RANK = int(os.getenv('RANK', -1))
ROOT = Path(os.path.dirname(__file__))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default = ROOT / 'data', help='data/val')
    parser.add_argument('--choice', default = 'torchvision-shufflenet_v2_x1_0', type=str)
    parser.add_argument('--num_classes', default = 6, type=int)
    parser.add_argument('--kwargs', default = "{}", type=str, )
    parser.add_argument('--weight', default = './run/exp/best.pt', help='configs for models, data, hyps')
    parser.add_argument('--transforms', default = 'to_tensor normalize', help='空格隔开')
    parser.add_argument('--imgsz', default = '[[720, 720], [360, 360]]',type=str, help='centercrop_resize resize center_crop')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    return parser.parse_args()

def main(opt):
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data
    data_cfgs = {}
    data_cfgs['root'] = opt.root
    data_cfgs['imgsz'] = eval(opt.imgsz)
    data_cfgs['val'] = {'augment': opt.transforms}

    data_processor = SmartDataProcessor(data_cfgs=data_cfgs, rank=RANK, project=None)
    dataset = data_processor.create_dataset('val')
    dataloader = data_processor.set_dataloader(dataset) # batchsize default 256

    # model
    model_cfg = {}
    model_cfg['choice'] = opt.choice
    model_cfg['num_classes'] = opt.num_classes
    model_cfg['kwargs'] = eval(opt.kwargs)
    model_cfg['pretrained'] = False
    model_cfg['backbone_freeze'] = False
    model_cfg['bn_freeze'] = False
    model_cfg['bn_freeze_affine'] = False

    model_processor = SmartModel(model_cfg)
    model = model_processor.model
    weights = torch.load(opt.weight, map_location=device)['model']
    model.load_state_dict(weights)
    model.to(device)

    # logger
    logger = SmartLogger()

    # val
    val(model, dataloader, device, None, False, None, logger)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
