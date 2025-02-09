from transformers import PreTrainedModel, PretrainedConfig
from transformers.processing_utils import ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature
from transforms import create_AugTransforms
import timm
from typing import List, Union, Optional
from PIL import Image
import torch

class DoraemonConfig(PretrainedConfig):
    model_type = "doraemon_classifier"
    
    def __init__(
        self,
        **kwargs
    ):
        self.model_config = kwargs.pop("config", {})
        self.class_dict = kwargs.pop("class_dict", {})
        
        super().__init__(**kwargs)

class DoraemonImageProcessor(ProcessorMixin):
    attributes = []
    config_class = DoraemonConfig
    
    def __init__(self, config: Optional[DoraemonConfig] = None, **kwargs):
        super().__init__()
        self.config = config
        self.transforms = create_AugTransforms(config.model_config["data"]["val"]["augment"])

    def __call__(
        self, 
        images: Union[Image.Image, List[Image.Image]], 
        return_tensors: Optional[str] = "pt",
        **kwargs
    ) -> BatchFeature:
        if isinstance(images, Image.Image):
            images = [images]
            
        image_tensors = self.preprocess(images)
        
        return BatchFeature(
            data={
                "pixel_values": image_tensors
            },
            tensor_type=return_tensors
        )
    
    def preprocess(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        if isinstance(images, Image.Image):
            images = [images]

        image_tensors = []
        for image in images:
            if image.mode == 'P':
                image = image.convert('RGBA')
            image = image.convert('RGB')
            image_tensor = self.transforms(image)
            image_tensors.append(image_tensor)

        return torch.stack(image_tensors, dim=0)
    
    def postprocess(self, outputs, **kwargs):
        return outputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(config=config)

class DoraemonClassifier(PreTrainedModel):
    config_class = DoraemonConfig
    
    def __init__(self, config: Optional[DoraemonConfig] = None, **kwargs):
        super().__init__(config)
        model_config = config.model_config["model"]
        self.model = timm.create_model(model_config["name"], 
                                       num_classes=model_config["num_classes"],
                                       pretrained=False)
        weights = torch.load(model_config["load_from"], 
                             map_location="cpu", 
                             weights_only=False)['ema'].float().state_dict()
        self.model.load_state_dict(weights)
        self.model.eval()
    
    def forward(self, pixel_values):
        with torch.inference_mode():
            output = self.model(pixel_values)

        return output
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(config=config)
        

if __name__ == "__main__":
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("./", trust_remote_code=True)
    image = Image.open("/workspace/data/train/sku-0702303518/sku_id-0702303518-10_crop_0.jpg")
    tensor = processor(image).to("cuda")
    # print(tensor)
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("./", trust_remote_code=True)
    print(config.__class__)

    from transformers import AutoModel
    model = AutoModel.from_pretrained("./", trust_remote_code=True)
    model.to("cuda")
    # print(model)

    print(model(tensor["pixel_values"]))