import random
import torch
from torchvision.models import get_model
from PIL import Image
import torchvision.transforms as T
import argparse
from dataset.transforms import create_AugTransforms
import cv2
import numpy as np
import os

def parse_opt():
    parsers = argparse.ArgumentParser()
    parsers.add_argument('--video', default='./record/5000201400/4204742643555836267_0_2023-05-25-01-02-17_2023-05-25-01-04-01.mp4', type=str)
    parsers.add_argument('--pt', default='./best.pt', type=str)
    parsers.add_argument('--transforms', default='centercrop_resize to_tensor_without_div')
    parsers.add_argument('--imgsz', default='[[720, 720], [224, 224]]', type=str)
    parsers.add_argument('--output', default=False, type=bool)
    parsers.add_argument('--names', default='[0,2,4,6,7,8,10]', type=str)
    parsers.add_argument('--sample', default=0.5, type=float, help='retain ratio')
    parsers.add_argument('--fps', default=25, type=int, help='FPS')
    parsers.add_argument('--video_imgsz', default='[720, 1280]', type=str, help='h w')

    args = parsers.parse_args()
    return args

def image_process(frame: np.array, transforms: T.Compose):
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = Image.fromarray(img)
    return transforms(img).unsqueeze(0)

def main(opt):

    # variable
    video_path = opt.video
    weight_path = opt.pt
    transforms = opt.transforms
    imgsz = opt.imgsz
    is_output = opt.output
    names = eval(opt.names)
    sample = opt.sample
    fps = opt.fps
    video_imgsz = eval(opt.video_imgsz)

    if is_output:
        filename = f'{os.path.splitext(video_path)[0]}_new.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename, fourcc, 25, (video_imgsz[1], video_imgsz[0])) # width, height
    # image
    # 获得视频的格式
    videoCapture = cv2.VideoCapture(video_path)

    # model
    model = get_model('mobilenet_v2', width_mult = 0.25)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 7)
    weight = torch.load(weight_path, map_location='cpu')['model']
    model.load_state_dict(weight)
    # eval
    model.eval()
    success, frame = videoCapture.read()
    while success:
        if random.random() > sample:
            success, frame = videoCapture.read()
            continue
        image = image_process(frame, create_AugTransforms(transforms, eval(imgsz)))

        result = torch.nn.functional.softmax(model(image), dim=-1)[0]
        idxes = result.argsort(0, descending=True)

        text = '\n'.join(f'{result[j].item():.2f} {names[j]}' for j in idxes).split('\n')

        cv2.putText(frame, str(text), (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 0, 255), 2)

        if is_output: out.write(frame)
        else:
            cv2.imshow('windows', frame)  # 显示
            cv2.waitKey(int(1000 / int(fps)))  # 延迟
        success, frame = videoCapture.read()  # 获取下一帧

    videoCapture.release()
    if is_output: out.release()

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)