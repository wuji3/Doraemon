from typing import Tuple, Callable, Optional, List
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import SwinTransformer
from PIL.JpegImagePlugin import JpegImageFile
from PIL.Image import Image as ImageType
from torchvision.transforms import Compose
from dataset.transforms import SPATIAL_TRANSFORMS


from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit

# -----------------ClassifierOutputTarget用于指定targets 具体的类 模型在看什么---------------------- #
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# ---------------------------------------------------------------------------------------------- #

class ClassActivationMaper:

    METHODS = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    def __init__(self, model: nn.Module, method: str, device):
        self.model = model
        self.device = device

        target_layers, reshape_transform = self._create_target_layers_and_transform(model)

        # cam
        if method not in self.METHODS:
            raise Exception(f"Method {method} not implemented")
        if method == "ablationcam":
            self.cam = ClassActivationMaper.METHODS[method](model=model,
                                       target_layers=target_layers,
                                       use_cuda=device != torch.device('cpu'),
                                       reshape_transform=reshape_transform,
                                       ablation_layer=AblationLayerVit())
        else:
            self.cam = ClassActivationMaper.METHODS[method](model=model,
                                       target_layers=target_layers,
                                       use_cuda=device != torch.device('cpu'),
                                       reshape_transform=reshape_transform)

    def __call__(self,
                 image,
                 input_tensor: torch.Tensor,
                 transforms: Compose,
                 targets: Optional[List[ClassifierOutputTarget]] = None) -> np.array:
        grayscale_cam = self.cam(input_tensor=input_tensor,
                                 targets=targets,
                                 eigen_smooth=False,
                                 aug_smooth=False)

        grayscale_cam = grayscale_cam[0, :]

        if type(image) not in (JpegImageFile, ImageType): raise ValueError("只能输入PIL.Image读取的图")

        spatial_transforms = ClassActivationMaper.pickup_spatial_transforms(transforms)
        image = spatial_transforms(image)
        image = np.array(image, dtype=np.float32)

        cam_image = show_cam_on_image(image / 255, grayscale_cam)

        return cam_image

    def _create_target_layers_and_transform(self, model: nn.Module) -> Tuple[list, Callable]:
        if type(model) is SwinTransformer:
            return [model.features[-1][-1].norm2], lambda tensor: torch.permute(tensor, dims=[0, 3, 1, 2])
        else: # CNN暂未实现
            raise KeyError(f'{type(model)} 还没在仓库里支持')

    @staticmethod
    def pickup_spatial_transforms(transforms: Compose):
        sequence = []
        for t in transforms.transforms:
            if type(t) in SPATIAL_TRANSFORMS:
                sequence.append(t)

        return Compose(sequence)

