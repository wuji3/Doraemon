from PIL import Image
import cv2
import numpy as np
import torch.nn.functional as F
from utils.general import increment_path
from utils.augment import create_AugTransforms
from utils.logger import SmartLogger
from utils.plots import colorstr
from utils.datasets import PredictDatasets
from torch.utils.data import DataLoader
from models import SmartModel
import os
import argparse
from pathlib import Path
import torch
import glob
from utils.plots import Annotator
import json
import time
from functools import reduce
import platform
import shutil

RANK = int(os.getenv('RANK', -1))
ROOT = Path(os.path.dirname(__file__))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default = ROOT / 'data/val/a', help='data/val')
    parser.add_argument('--show_path', default = ROOT / 'visualization')
    parser.add_argument('--save_txt', action='store_true')
    parser.add_argument('--badcase', action='store_true')
    parser.add_argument('--is_cam', action='store_true')
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--name', default = 'exp')
    parser.add_argument('--choice', default = 'torchvision-shufflenet_v2_x1_0', type=str)
    parser.add_argument('--kwargs', default="{}", type=str, )
    parser.add_argument('--class_head', default = 'ce', type=str, help='ce or bce')
    parser.add_argument('--class_json', default = './run/exp/class_indices.json', type=str)
    parser.add_argument('--num_classes', default = 5, type=int, help='out channels of fc 训的时候可能多留几个神经元')
    parser.add_argument('--weight', default = './run/exp/best.pt', help='configs for models, data, hyps')
    parser.add_argument('--transforms', default = {'to_tensor': 'no_params', 'normalize': 'no_params'}, help='空格隔开')
    parser.add_argument('--attention_pool', action='store_true', help='是否使用注意力池化, 默认False, 即使用平均池化')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    return parser.parse_args()

def predict_images(model, root, visual_path, transforms, class_head: str, class_indices: dict, save_txt: bool, logger, device, badcase: bool, is_cam: bool):

    assert class_head in {'bce', 'ce'}
    os.makedirs(visual_path, exist_ok=True)

    dataset = PredictDatasets(root,
                              transforms=create_AugTransforms(eval(transforms) if isinstance(transforms, str) else transforms),
                              postfix='jpg')
    dataloader = DataLoader(dataset, shuffle=False, pin_memory=True, num_workers=2, batch_size=1, collate_fn=PredictDatasets.collate_fn)

    # if not imgs_path: raise FileExistsError(f'root不含图像')
    # eval mode
    model.eval()
    n = len(dataloader)

    # cam
    if is_cam:
        from utils.cam import ClassActivationMaper
        cam = ClassActivationMaper(model, method='gradcam', device=device)

    for i, (img, inputs, img_path) in enumerate(dataloader):
        img = img[0]
        img_path = img_path[0]

        if is_cam:
            cam_image = cam(image=img, input_tensor=inputs, transforms=dataset.transforms)
            # cam_image = Image.fromarray(cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB).astype('uint8'))
            # cam_image = cam_image.resize(img.size, resample=2)
            cam_image = cv2.resize(cam_image, img.size, interpolation=cv2.INTER_LINEAR)

        # system
        if platform.system().lower() == 'windows':
            annotator = Annotator(img, font=r'C:/WINDOWS/FONTS/SIMSUN.TTC') # windows
        else:
            annotator = Annotator(img) # linux

        # transforms
        inputs = inputs.to(device)
        # forward
        logits = model(inputs).squeeze()

        # post process
        if class_head == 'ce':
            probs = F.softmax(logits, dim=0)
        else:
            probs = F.sigmoid(logits)

        top5i = probs.argsort(0, descending=True)[:5].tolist()

        text = '\n'.join(f'{probs[j].item():.2f} {class_indices[j]}' for j in top5i)

        annotator.text((32, 32), text, txt_color=(0, 0, 0))
        if save_txt:  # Write to file
            os.makedirs(os.path.join(visual_path, 'labels'), exist_ok=True)
            with open(os.path.join(visual_path, 'labels', os.path.basename(img_path).replace('.jpg','.txt')), 'a') as f:
                f.write(text + '\n')

        logger.console(f"[{i+1}|{n}] " + os.path.basename(img_path) +" " + reduce(lambda x,y: x + " "+ y, text.split()))

        if is_cam:
            img = np.hstack([cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), cam_image])
            cv2.imwrite(os.path.join(visual_path, os.path.basename(img_path)), img)
        else: img.save(os.path.join(visual_path, os.path.basename(img_path)))

    if badcase:
        cls = root.split('/')[-1]
        os.makedirs(os.path.join(visual_path, 'bad_case'), exist_ok=True)
        for txt in glob.glob(os.path.join(visual_path, 'labels', '*.txt')):
            with open(txt, 'r') as f:
                if f.readlines()[0].split()[1] != cls:
                    try:
                        shutil.move(os.path.join(visual_path, os.path.basename(txt).replace('.txt', '.jpg')), os.path.dirname(txt).replace('labels','bad_case'))
                    except FileNotFoundError:
                        print(f'FileNotFoundError->{txt}')
def main(opt):

    visual_dir = increment_path(Path(opt.show_path) / opt.name)

    with open(opt.class_json, 'r', encoding='utf-8') as f:
        class_dict = json.load(f)
        class_dict = dict((eval(k), v) for k,v in class_dict.items())

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model -> do not modify
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

    # predict
    t0 = time.time()
    predict_images(model, opt.root, visual_dir, opt.transforms, opt.class_head, class_dict, opt.save_txt, logger, device, opt.badcase, opt.is_cam)


    logger.console(f'\nPredicting complete ({(time.time() - t0) / 60:.3f} minutes)'
                   f"\nResults saved to {colorstr('bold', visual_dir)}")

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
