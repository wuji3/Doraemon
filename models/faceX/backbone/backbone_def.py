import yaml
from .resnets import Resnet
from .swin_transformer import SwinTransformer
from .efficientnets import efficientnet, EfficientNet


class BackboneFactory:
    """Factory to produce backbone according the backbone_conf.yaml.
    
    Attributes:
        backbone_type(str): which backbone will produce.
        backbone_param(dict):  params about model structure.
    """

    def __init__(self, backbone_config):
        # self.backbone_type = list(backbone_config['backbone'].keys())[0]
        # self.backbone_param = backbone_config['backbone'][self.backbone_type]
        for k, v in backbone_config.items(): self.backbone_type, self.backbone_param = k, v

    def get_backbone(self):

        if self.backbone_type == 'resnet':
            depth = self.backbone_param['depth'] # depth of the ResNet, e.g. 50, 100, 152.
            drop_ratio = self.backbone_param['drop_ratio'] # drop out ratio.
            net_mode = self.backbone_param['net_mode'] # 'ir' for improved by resnt, 'ir_se' for SE-ResNet.
            feat_dim = self.backbone_param['feat_dim'] # dimension of the output features, e.g. 512.
            out_h = self.backbone_param['out_h'] # height of the feature map before the final features.
            out_w = self.backbone_param['out_w'] # width of the feature map before the final features.

            backbone = Resnet(depth, drop_ratio, net_mode, feat_dim, out_h, out_w)

        elif self.backbone_type == 'efficientnet':
            width = self.backbone_param['width'] # width for EfficientNet, e.g. 1.0, 1.2, 1.4, ...
            depth = self.backbone_param['depth'] # depth for EfficientNet, e.g. 1.0, 1.2, 1.4, ...
            image_size = self.backbone_param['image_size'] # input image size, e.g. 112.
            drop_ratio = self.backbone_param['drop_ratio'] # drop out ratio.
            out_h = self.backbone_param['out_h'] # height of the feature map before the final features.
            out_w = self.backbone_param['out_w'] # width of the feature map before the final features.
            feat_dim = self.backbone_param['feat_dim'] # dimension of the output features, e.g. 512.
            blocks_args, global_params = efficientnet(
                width_coefficient=width, depth_coefficient=depth,
                dropout_rate=drop_ratio, image_size=image_size)

            backbone = EfficientNet(out_h, out_w, feat_dim, blocks_args, global_params)

        elif self.backbone_type == 'swin transformer':
            img_size = self.backbone_param['img_size']
            patch_size= self.backbone_param['patch_size']
            in_chans = self.backbone_param['in_chans']
            embed_dim = self.backbone_param['embed_dim']
            depths = self.backbone_param['depths']
            num_heads = self.backbone_param['num_heads']
            window_size = self.backbone_param['window_size']
            mlp_ratio = self.backbone_param['mlp_ratio']
            drop_rate = self.backbone_param['drop_rate']
            drop_path_rate = self.backbone_param['drop_path_rate']

            backbone = SwinTransformer(img_size=img_size,
                                       patch_size=patch_size,
                                       in_chans=in_chans,
                                       embed_dim=embed_dim,
                                       depths=depths,
                                       num_heads=num_heads,
                                       window_size=window_size,
                                       mlp_ratio=mlp_ratio,
                                       qkv_bias=True,
                                       qk_scale=None,
                                       drop_rate=drop_rate,
                                       drop_path_rate=drop_path_rate,
                                       ape=False,
                                       patch_norm=True,
                                       use_checkpoint=False)

        else:
            raise NotImplemented("only resnet, efficientnet and swin transformer are supported now !")

        return backbone
