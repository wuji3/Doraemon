from .resnets import Resnet
# from .swin_transformer import SwinTransformer
from .swin import SwinTransformer
from .efficientnets import efficientnet, EfficientNet
from .convnext import ConvNeXt

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

        if self.backbone_type == 'resnet':
            image_size = self.backbone_param['image_size']
            depth = self.backbone_param['depth'] # depth of the ResNet, e.g. 50, 100, 152.
            drop_ratio = self.backbone_param['drop_ratio'] # drop out ratio.
            net_mode = self.backbone_param['net_mode'] # 'ir' for improved by resnt, 'ir_se' for SE-ResNet.
            feat_dim = self.backbone_param['feat_dim'] # dimension of the output features, e.g. 512.

            backbone = Resnet(num_layers=depth, 
                              image_size=image_size,
                              drop_ratio=drop_ratio, 
                              mode=net_mode, 
                              feat_dim=feat_dim)

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
            model_size = self.backbone_param['model_size']
            img_size = self.backbone_param['image_size']
            feat_dim = self.backbone_param['feat_dim'] # dimension of the output features, e.g. 512.

            backbone = SwinTransformer(img_size = img_size, model_size = model_size, feat_dim = feat_dim)

        elif self.backbone_type == 'convnext':
            model_size = self.backbone_param['model_size']
            feat_dim = self.backbone_param['feat_dim']
            image_size = self.backbone_param['image_size']

            backbone = ConvNeXt(model_size = model_size,
                                feat_dim=feat_dim,
                                image_size=image_size)
        else:
            raise NotImplemented("only resnet, efficientnet and swin transformer are supported now !")

        return backbone
