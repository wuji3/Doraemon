import shutil
import torch
from torch.distributed import init_process_group
from doraemon import (
    CenterProcessor,
    yaml_load,
    increment_path,
    DistillCenterProcessor,
    colorstr,
    check
)
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
    parser.add_argument('cfgs', default = ROOT / 'configs/classification/pet.yaml', help='configs for models, data, hyps')
    parser.add_argument('--resume', default = '', help='if no resume, not write')
    parser.add_argument('--sync_bn', action='store_true', help='turn on syncBN, if on, speed will be slower')
    parser.add_argument('--project', default=ROOT / 'run', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--distill', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    # face/cbir
    parser.add_argument('--print_freq', type=int, default=50, help='The print frequency for training state')
    parser.add_argument('--save_freq', type=int, default=5, help='The checkpoint frequency for saving state_dict epoch-wise, not contains warm epochs')
    return parser.parse_args()

def main(opt):
    save_dir = increment_path(Path(opt.project) / opt.name)
    opt.save_dir = save_dir

    assert torch.cuda.device_count() > LOCAL_RANK
    # init process groups
    if LOCAL_RANK != -1:
        init_process_group(backend='nccl', world_size = WORLD_SIZE, rank = LOCAL_RANK)

    # configs
    cfgs = yaml_load(opt.cfgs)
    task: str= cfgs['model'].get('task', None)

    # check configs
    check(task, cfgs)

    # init cpu
    cpu = CenterProcessor(cfgs, LOCAL_RANK, project=save_dir, opt=opt) \
        if not opt.distill else DistillCenterProcessor(cfgs, LOCAL_RANK, project=save_dir, opt=opt)

    # record config
    shutil.copy(opt.cfgs, save_dir)

    # syncBN
    if LOCAL_RANK != -1 and opt.sync_bn:
        cpu.set_sync_bn()
        if LOCAL_RANK == 0:
            cpu.logger.both(f'{colorstr("yellow", "Attention")}: sync_bn is on')
    # run
    cpu.run_classifier(resume=opt.resume if opt.resume else None) \
        if task == 'classification' else cpu.run_embedding(resume=opt.resume if opt.resume else None)

if __name__ == '__main__':
    opts = parse_opt()
    main(opts)