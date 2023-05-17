import onnxruntime
from PIL import Image
import torchvision.transforms as T
import argparse
from utils.augment import create_AugTransforms

def parse_opt():
    parsers = argparse.ArgumentParser()

    parsers.add_argument('--img', default='./img.png', type=str)
    parsers.add_argument('--onnx', default='./shufflev2_0.5.onnx', type=str)
    parsers.add_argument('--transforms', default='centercrop_resize to_tensor normalize')
    parsers.add_argument('--imgsz', default='[[720, 720], [360, 360]]', type=str)
    parsers.add_argument('--input_onnx', default='input', type=str, help = 'input_name of onnx ')
    parsers.add_argument('--output_onnx', default='prob', type=str, help = 'output_name of onnx ')

    args = parsers.parse_args()
    return args

def image_process(path: str, transforms: T.Compose):
    img = Image.open(path).convert('RGB')
    return transforms(img).unsqueeze(0).numpy()

def main(opt):

    # variable
    img_path = opt.img
    onnx_path = opt.onnx
    transforms = opt.transforms
    imgsz = opt.imgsz
    intput_name = opt.input_onnx
    output_name = opt.output_onnx

    # image
    image = image_process(img_path, create_AugTransforms(transforms, eval(imgsz)))

    # model
    session = onnxruntime.InferenceSession(onnx_path)
    output = session.run([f'{output_name}'], {f'{intput_name}': image})[0]

    print(output)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

