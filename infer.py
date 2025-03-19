from engine import increment_path
from dataset.transforms import create_AugTransforms
from utils.plots import colorstr
from dataset.basedataset import PredictImageDatasets
from torch.utils.data import DataLoader
import os
import argparse
from pathlib import Path
import torch
import time
from engine import Visualizer
from models import get_model
from utils.logger import SmartLogger
from models.faceX.face_model import FaceModelLoader
from engine.cbir.evaluation import valuate as valuate_cbir
from tqdm import tqdm

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
ROOT = Path(os.path.dirname(__file__))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('model-path', default='run/exp8', help='Path to model configs')

    # classification
    parser.add_argument('--data', default = ROOT / 'data', help='Target data directory')
    parser.add_argument('--infer-option', choices=['default', 'autolabel'], default='default', 
                        help='default: Infer + Visualize + CaseAnalysis, autolabel: Infer + Label')
    parser.add_argument('--split', default=None, type=str, help='Split to visualize')
    parser.add_argument('--classes', default=None, nargs='+', help='Which class to check')
    parser.add_argument('--ema', action='store_true', help = 'Exponential Moving Average for model weight')
    parser.add_argument('--sampling', default=None, type=int, help='Sample n images for visualization')

    # CBIR
    parser.add_argument('--max_rank', default=10, type=int, help='Visualize top k retrieval results')
    parser.add_argument('--root', default = None, help = 'Prediction root path for cbir dataset (If need change from cfgs)')

    # Unless specific needs, it is generally not modified below.
    parser.add_argument('--show_path', default = ROOT / 'inference')
    parser.add_argument('--name', default = 'exp')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    return parser.parse_args()

if __name__ == '__main__':
    if LOCAL_RANK != -1:
        device = torch.device('cuda', LOCAL_RANK)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    opt = parse_opt()
    visual_dir = increment_path(Path(opt.show_path) / opt.name)

    pt_file = Path(getattr(opt, 'model-path'))
    pt = torch.load(pt_file, weights_only=False)
    config = pt['config']
    task: str = config['model']['task']    

    logger = SmartLogger(filename=None, level=1) if LOCAL_RANK in {-1,0} else None

    if task == 'classification':
        modelwrapper = get_model(config['model'], None, LOCAL_RANK)
        model = modelwrapper.load_weight(pt, ema=opt.ema, device=device)
        
        if opt.classes is None:
            opt.classes = list(pt['label2id'])
            
        dataset = PredictImageDatasets(opt.data,
                                       transforms=create_AugTransforms(config['data']['val']['augment']), 
                                       sampling=opt.sampling, 
                                       classes=opt.classes,
                                       split=opt.split,
                                       require_gt= opt.infer_option == 'default')
        dataloader = DataLoader(dataset, 
                              shuffle=False, 
                              pin_memory=True, 
                              num_workers=config['data']['nw'], 
                              batch_size=1,
                              collate_fn=PredictImageDatasets.collate_fn)

        t0 = time.time()
        Visualizer.predict_images(model,
                                dataloader,
                                device,
                                visual_dir,
                                pt['id2label'],
                                logger,
                                config['hyp']['loss']['bce'][1] if config['hyp']['loss']['bce'][0] else 0,
                                opt.infer_option
                                )

        logger.console(f'\nPredicting complete ({(time.time() - t0) / 60:.3f} minutes)'
                    f"\nResults saved to {colorstr('bold', visual_dir)}")
    elif task in ('face', 'cbir'):
        # logger
        logger = SmartLogger(filename=None)

        # checkpoint loading
        logger.console(f'Loading Model, EMA Is {opt.ema}')
        model_loader = FaceModelLoader(model_cfg=config['model'])
        model = model_loader.load_weight(model_path=pt_file, ema=opt.ema)

        if opt.root is not None:
            config['data']['root'] = opt.root

        config['data']['val']['metrics']['cutoffs'] = [opt.max_rank]
        metrics, retrieval_results, scores, ground_truths, queries, query_dataset, gallery_dataset = valuate_cbir(model, 
                                                                         config['data'], 
                                                                         device,
                                                                         logger, 
                                                                         vis=True)

        for idx, q in tqdm(enumerate(queries), total=len(queries), desc='Visualizing', position=0):
            Visualizer.visualize_results(q, 
                                         retrieval_results[idx], 
                                         scores[idx], 
                                         ground_truths[idx],
                                         visual_dir,
                                         opt.max_rank,
                                         query_dataset,
                                         gallery_dataset
                                         )

        logger.console(f'Metrics: {metrics}')

    else:
        raise ValueError(f'Unknown task {task}')