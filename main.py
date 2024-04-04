import shutil
import torch
from torch.distributed import init_process_group
from engine.vision_engine import CenterProcessor, yaml_load, increment_path, check_cfgs
from utils.plots import colorstr
from distills import DistillCenterProcessor
import os
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(os.path.dirname(__file__))
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgs', default = ROOT / 'configs/complete.yaml', help='configs for models, data, hyps')
    parser.add_argument('--resume', default = '', help='if no resume, not write')
    parser.add_argument('--load_from', default = '', help='load weight for finetune')
    parser.add_argument('--sync_bn', action='store_true', help='turn on syncBN, if on, speed will be slower')
    parser.add_argument('--project', default=ROOT / 'run', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--distill', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    return parser.parse_args()

def main(opt):
    save_dir = increment_path(Path(opt.project) / opt.name)
    assert torch.cuda.device_count() > LOCAL_RANK
    # init process groups
    if LOCAL_RANK != -1:
        init_process_group(backend='nccl', world_size = WORLD_SIZE, rank = LOCAL_RANK)
    # configs
    cfgs = yaml_load(opt.cfgs)
    check_cfgs(cfgs)
    # add load_from if need
    if opt.load_from:  cfgs['model']['load_from'] = opt.load_from
    # init cpu
    cpu = CenterProcessor(cfgs, LOCAL_RANK, project=save_dir) if not opt.distill else DistillCenterProcessor(cfgs, LOCAL_RANK, project=save_dir)
    # record config
    shutil.copy(opt.cfgs, save_dir)
    cpu.cfg_path = opt.cfgs
    # syncBN
    if LOCAL_RANK != -1 and opt.sync_bn:
        cpu.set_sync_bn()
        if LOCAL_RANK == 0:
            cpu.logger.both(f'{colorstr("yellow", "Attention")}: sync_bn is on')
    # run
    cpu.run(resume=opt.resume if opt.resume else None)

if __name__ == '__main__':
    opts = parse_opt()
    main(opts)