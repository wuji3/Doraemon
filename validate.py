import os
from os.path import join as opj
import argparse
from pathlib import Path
from engine import valuate, CenterProcessor, yaml_load
import torch

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
ROOT = Path(os.path.dirname(__file__))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgs', default = 'run/exp/pet.yaml', help = 'Configs for models, data, hyps')
    parser.add_argument('--weight', default = 'run/exp/best.pt', help='Weight path')
    parser.add_argument('--eval_topk', default = 5, type=int, help = 'Tell topk_acc, maybe top5, top3...')
    parser.add_argument('--ema', action='store_true',help = 'Exponential Moving Average for model weight')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    return parser.parse_args()

def main(opt):

    cfgs = yaml_load(opt.cfgs)
    cpu = CenterProcessor(cfgs, LOCAL_RANK, train=False)

    # checkpoint loading
    model = cpu.model_processor.model
    if opt.ema:
        weights = torch.load(opt.weight, map_location=cpu.device)['ema'].float().state_dict()
    else:
        weights = torch.load(opt.weight, map_location=cpu.device)['model']
    model.load_state_dict(weights)

    # set val dataloader
    dataloader = cpu.data_processor.set_dataloader(cpu.data_processor.val_dataset, nw=cpu.data_cfg['nw'], bs=cpu.data_cfg['train']['bs'],
                                                   collate_fn=cpu.data_processor.val_dataset.collate_fn)

    conm_path = opj(os.path.dirname(opt.weight), 'conm.png')
    valuate(model, dataloader, cpu.device, None, False, None, cpu.logger, thresh=cpu.thresh, top_k=opt.eval_topk,
            conm_path=conm_path)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)