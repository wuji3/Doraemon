from .resnets import Resnet
# from .swin_transformer import SwinTransformer
from .swin import SwinTransformer
from .efficientnets import efficientnet, EfficientNet
from .convnext import ConvNeXt
from .timm_wrapper import TimmWrapper

class BackboneFactory:
    """Factory to produce backbone according the backbone_conf.yaml.
    
    Attributes:
        backbone_type: which backbone will produce.
        backbone_param:  params about model structure.
    """

    def __init__(self, backbone_config):
        # self.backbone_type = list(backbone_config['backbone'].keys())[0]
        # self.backbone_param = backbone_config['backbone'][self.backbone_type]
        for k, v in backbone_config.items(): self.backbone_type, self.backbone_param = k, v

    def get_backbone(self):

        if self.backbone_type.startswith('timm'):
            model = self.backbone_type.split('-')[1]
            return TimmWrapper(
                model_name=model,
                **self.backbone_param,
            )

        elif self.backbone_type.startswith('torchvision'):  # torchvision
            raise NotImplementedError("Torchvision models are not supported yet.")

        if self.backbone_type == 'resnet':

            backbone = Resnet(**self.backbone_param)

        elif self.backbone_type == 'efficientnet':
            width = self.backbone_param['width'] # width for EfficientNet, e.g. 1.0, 1.2, 1.4, ...
            depth = self.backbone_param['depth'] # depth for EfficientNet, e.g. 1.0, 1.2, 1.4, ...
            image_size = self.backbone_param['image_size'] # input image size, e.g. 112.
            drop_ratio = self.backbone_param['drop_ratio'] # drop out ratio.
            image_size = self.backbone_param['image_size']
            feat_dim = self.backbone_param['feat_dim'] # dimension of the output features, e.g. 512.
            blocks_args, global_params = efficientnet(
                width_coefficient=width, depth_coefficient=depth,
                dropout_rate=drop_ratio, image_size=image_size)

            backbone = EfficientNet(orj_image_size=image_size, 
                                    feat_dim=feat_dim, 
                                    blocks_args=blocks_args, 
                                    global_params=global_params)

        elif self.backbone_type == 'swintransformer':

            backbone = SwinTransformer(**self.backbone_param)

        elif self.backbone_type == 'convnext':

            backbone = ConvNeXt(**self.backbone_param)

        else:
            raise NotImplemented(f"{self.backbone_type} is not supported now !")

        return backbone
