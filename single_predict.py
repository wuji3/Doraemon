import torch
from torchvision.models import get_model
from PIL import Image
import torchvision.transforms as T
import argparse
from dataset.transforms import create_AugTransforms

def parse_opt():
    parsers = argparse.ArgumentParser()

    parsers.add_argument('--img', default='./edd4dc86b50997f29b81ba0b2bab1906.jpg', type=str)
    parsers.add_argument('--pt', default='./best.pt', type=str)
    parsers.add_argument('--transforms', default={'resize': [640, 640], 'to_tensor': 'no_params', 'normalize': 'no_params'})

    args = parsers.parse_args()
    return args

def image_process(path: str, transforms: T.Compose):
    img = Image.open(path).convert('RGB')
    return transforms(img).unsqueeze(0)

def main(opt):

    # variable
    img_path = opt.img
    weight_path = opt.pt
    transforms = opt.transforms

    # image
    image = image_process(img_path, create_AugTransforms(eval(transforms) if isinstance(transforms, str) else transforms))

    # model
    model = get_model('mobilenet_v2', width_mult = 0.25)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 7)
    weight = torch.load(weight_path, map_location='cpu')['model']
    model.load_state_dict(weight)
    # eval
    model.eval()

    out = model(image)
    print(torch.nn.functional.softmax(out, dim=-1))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)