import os
from os.path import join as opj
import argparse
from pathlib import Path
from engine import valuate as valuate_classifier
import torch
from models.faceX.face_model import FaceModelLoader
from engine.faceX.evaluation import valuate as valuate_face
from engine.cbir.evaluation import valuate as valuate_cbir
from prettytable import PrettyTable
from utils.logger import SmartLogger
from models import get_model
from dataset.dataprocessor import SmartDataProcessor

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
ROOT = Path(os.path.dirname(__file__))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('model-path', default='run/exp/best.pt', help='Path to model configs')
    parser.add_argument('--ema', action='store_true', help='Exponential Moving Average for model weight')
    
    # classifier
    parser.add_argument('--eval_topk', default=5, type=int, help='Tell topk_acc, maybe top5, top3...')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    
    return parser.parse_args()

def main(opt):
    # device
    if LOCAL_RANK != -1:
        device = torch.device('cuda', LOCAL_RANK)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger = SmartLogger(filename=None, level=1) if LOCAL_RANK in {-1,0} else None

    pt_file = Path(getattr(opt, 'model-path'))
    pt = torch.load(pt_file, weights_only=False)
    config = pt['config']
    task: str = config['model']['task']

    if task == 'classification':
        modelwrapper = get_model(config['model'], None, LOCAL_RANK)
        # checkpoint loading
        model = modelwrapper.load_weight(pt, ema=opt.ema, device=device)

        data_processor = SmartDataProcessor(config['data'], LOCAL_RANK, None, training = False)
        data_processor.val_dataset = data_processor.create_dataset('val', training = False, id2label=pt['id2label']) 

        # set val dataloader
        dataloader = data_processor.set_dataloader(data_processor.val_dataset, nw=config['data']['nw'], bs=config['data']['val']['bs'],
                                                       collate_fn=data_processor.val_dataset.collate_fn)

        conm_path = opj(os.path.dirname(pt_file), 'conm.png')
        
        thresh = config['hyp']['loss']['bce'][1] if config['hyp']['loss']['bce'][0] else 0
        valuate_classifier(model, dataloader, device, None, False, None, logger, thresh=thresh, top_k=opt.eval_topk,
                conm_path=conm_path)

    elif task in ('face', 'cbir'):
        # logger
        logger = SmartLogger(filename=None)

        # checkpoint loading
        logger.console(f'Loading Model, EMA is {opt.ema}')
        model_loader = FaceModelLoader(model_cfg=config['model'])
        model = model_loader.load_weight(model_path=pt_file, ema=opt.ema)

        logger.console('Evaluating...')
        if task == 'face':
            mean, std = valuate_face(model, config['data'], torch.device('cuda'))
            pretty_tabel = PrettyTable(["model_name", "mean accuracy", "standard error"])
            pretty_tabel.add_row([os.path.basename(pt_file), mean, std])

            logger.console('\n' + str(pretty_tabel))
        else:
            metrics = valuate_cbir(model, 
                                   config['data'], 
                                   torch.device('cuda', LOCAL_RANK if LOCAL_RANK > 0 else 0), 
                                   logger,
                                   vis=False)
            logger.console(metrics)

    else:
        raise ValueError(f'Unknown task {task}')

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)