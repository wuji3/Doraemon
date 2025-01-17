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

        else:
            raise NotImplemented(f"{self.backbone_type} is not supported now !")