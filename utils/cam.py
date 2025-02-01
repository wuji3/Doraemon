from typing import Tuple, Callable, Optional, List
import numpy as np
import torch
import torch.nn as nn

from timm.models import VisionTransformer, \
    SwinTransformer, \
    ResNet, \
    MobileNetV3, \
    ConvNeXt, \
    SwinTransformerV2, \
    SENet, \
    EfficientNet

from PIL.JpegImagePlugin import JpegImageFile
from PIL.Image import Image as ImageType
from torchvision.transforms import Compose
from dataset.transforms import SPATIAL_TRANSFORMS
from dataset.transforms import PadIfNeed, Reverse_PadIfNeed, ResizeAndPadding2Square, ReverseResizeAndPadding2Square
import cv2

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

# -----------------ClassifierOutputTarget is used to specify the targets, specifically which class the model is looking at---------------------- #
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

    def __init__(self, model: nn.Module, method: str, device, transforms):
        self.model = model
        self.device = device

        target_layers, reshape_transform = self._create_target_layers_and_transform(model)

        # cam
        if method not in self.METHODS:
            raise Exception(f"Method {method} not implemented")
        if method == "ablationcam":
            self.cam = ClassActivationMaper.METHODS[method](model=model,
                                       target_layers=target_layers,
                                       reshape_transform=reshape_transform,
                                       use_cuda = device != torch.device('cpu'),
                                       ablation_layer=AblationLayerVit())
        else:
            self.cam = ClassActivationMaper.METHODS[method](model=model,
                                       target_layers=target_layers,
                                       use_cuda = device != torch.device('cpu'),
                                       reshape_transform=reshape_transform)

        self.spatial_transforms, reversed_fun = ClassActivationMaper.pickup_spatial_transforms(transforms)

        if reversed_fun is not None:
            self.reverse_pad2square = reversed_fun

    def __call__(self,
                 image,
                 input_tensor: torch.Tensor,
                 dsize: Tuple[int, int],# w, h
                 targets: Optional[List[ClassifierOutputTarget]] = None) -> np.array:
        grayscale_cam = self.cam(input_tensor=input_tensor,
                                 targets=targets,
                                 eigen_smooth=False,
                                 aug_smooth=False)

        grayscale_cam = grayscale_cam[0, :]

        if type(image) not in (JpegImageFile, ImageType): raise ValueError("Only images read by PIL.Image are allowed")

        image = self.spatial_transforms(image)
        image = np.array(image, dtype=np.float32)

        cam_image = show_cam_on_image(image / 255, grayscale_cam)

        if hasattr(self, 'reverse_pad2square'):
            if max(dsize) != max(cam_image.shape):
                if isinstance(self.reverse_pad2square, Reverse_PadIfNeed):
                    cam_image = cv2.resize(cam_image, (max(dsize), max(dsize)), cv2.INTER_LINEAR)
                elif isinstance(self.reverse_pad2square, ReverseResizeAndPadding2Square):
                    pass
                else:
                    raise ValueError(f"{type(self.reverse_pad2square)} not support reverse function")
                cam_image = self.reverse_pad2square(cam_image, dsize)
        return cam_image

    def _create_target_layers_and_transform(self, model: nn.Module) -> Tuple[list, Optional[Callable]]:

        if isinstance(model, (SwinTransformer, SwinTransformerV2)):
            return [model.norm], lambda tensor: torch.permute(tensor, dims=[0, 3, 1, 2])

        elif isinstance(model, VisionTransformer):
            # Get patch and image size information
            patch_size = model.patch_embed.patch_size
            if isinstance(patch_size, int):
                patch_size = (patch_size, patch_size)
            
            img_size = model.patch_embed.img_size
            if isinstance(img_size, int):
                img_size = (img_size, img_size)
            
            # Calculate feature map size
            feature_size = (img_size[0] // patch_size[0],
                           img_size[1] // patch_size[1])
            
            def reshape_transform(tensor):
                # Remove CLS token
                tensor = tensor[:, 1:, :]
                
                # Reshape to [batch_size, height, width, channels]
                B, _, C = tensor.shape
                H, W = feature_size
                tensor = tensor.reshape(B, H, W, C)
                
                # Convert to [batch_size, channels, height, width]
                tensor = tensor.permute(0, 3, 1, 2)
                return tensor
            
            return [model.blocks[-1].norm1], reshape_transform

        elif isinstance(model, MobileNetV3):
            return [model.blocks[-1][0].conv], None

        elif isinstance(model, (SENet, ResNet)):
            return [model.layer4[-1].conv3], None

        elif isinstance(model, ConvNeXt):
            return [model.norm_pre], None

        elif isinstance(model, EfficientNet):
            return [model.bn2], None

        else:
            raise KeyError(f'{type(model)} not support yet')

    @staticmethod
    def pickup_spatial_transforms(transforms: Compose):
        sequence = []
        reversed_fun = None
        for t in transforms.transforms:
            if type(t) in SPATIAL_TRANSFORMS:
                sequence.append(t)
            if type(t) is PadIfNeed:
                reversed_fun = Reverse_PadIfNeed(mode=t.mode)
            elif type(t) is ResizeAndPadding2Square:
                reversed_fun = ReverseResizeAndPadding2Square(size=t.size)

        return Compose(sequence), reversed_fun