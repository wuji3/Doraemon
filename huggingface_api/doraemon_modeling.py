from transformers import PreTrainedModel, PretrainedConfig
from transformers.processing_utils import ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature
from .transforms import create_AugTransforms
from .model import SwinTransformer
from typing import List, Union, Optional
from PIL import Image
import torch
import requests
from io import BytesIO
from huggingface_hub import hf_hub_download
import torch.nn.functional as F

class DoraemonConfig(PretrainedConfig):
    model_type = "doraemon_embedding"
    
    def __init__(
        self,
        **kwargs
    ):
        self.model_config = kwargs.pop("config", {})
        
        super().__init__(**kwargs)

class DoraemonImageProcessor(ProcessorMixin):
    attributes = []
    config_class = DoraemonConfig
    
    def __init__(self, config: Optional[DoraemonConfig] = None, **kwargs):
        super().__init__()
        self.config = config
        self.transforms = create_AugTransforms(config.model_config["augment"])

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
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(config=config)

class DoraemonEmbedding(PreTrainedModel):
    config_class = DoraemonConfig
    
    def __init__(self, config: Optional[DoraemonConfig] = None, **kwargs):
        super().__init__(config)
        self.model = SwinTransformer(model_size='base')
        
        # Get the repo_id and filename from config or kwargs
        repo_id = kwargs.get('pretrained_model_name_or_path', 'srpone/im-embedding')
        filename = config.model_config.get("load_from", "Epoch_26.pt")
        revision = kwargs.get('revision', 'v4')
        
        # Download weights from the Hugging Face repository
        weights_path = hf_hub_download(
            revision=revision,
            repo_id=repo_id,
            filename=filename,
            cache_dir=None,
            force_download=False
        )
        
        weights = torch.load(weights_path, 
                           map_location="cpu", 
                           weights_only=False)['ema']
        self.model.load_state_dict(weights, strict=True)
        self.model.eval()
    
    def forward(self, pixel_values):
        with torch.inference_mode():
            output = self.model(pixel_values)
            output = F.normalize(output, dim=1)
            output = output.cpu().tolist()

        return output
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(config=config)
        

if __name__ == "__main__":
    from transformers import AutoProcessor

    # 使用URL加载图像
    # image_url = "https://m.media-amazon.com/images/I/61LStHWtbNL._AC_SY879_.jpg"  # 替换为实际的图像URL
    image_url = "https://m.media-amazon.com/images/I/817C+Gi-6uL._AC_SY879_.jpg"
    # image_url = "https://m.media-amazon.com/images/I/71b9rur70QL._AC_SY879_.jpg"
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    print("原始图像尺寸:", image.size)
    
    # 根据指定的坐标裁剪图像 [x1, y1, x2, y2]
    # crop_box = [102, 192, 520, 802]  # xyxy 格式
    # cropped_image = image.crop(crop_box)
    # print("裁剪后图像尺寸:", cropped_image.size)
    
    # 使用裁剪后的图像进行推理
    processor = AutoProcessor.from_pretrained("./", trust_remote_code=True)
    tensor = processor(image).to("cuda")
    
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("./", trust_remote_code=True)
    print(config.__class__)

    from transformers import AutoModel
    model = AutoModel.from_pretrained("./", trust_remote_code=True)
    model.to("cuda")

    print(model(tensor["pixel_values"]))