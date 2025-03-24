from transformers import PreTrainedModel, PretrainedConfig
from transformers.processing_utils import ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature
from typing import Optional
import torch
from huggingface_hub import hf_hub_download
from doraemon import create_AugTransforms
from timm import create_model
import torch.nn.functional as F
import numpy as np
import os

class DoraemonConfig(PretrainedConfig):
    model_type = "doraemon" 
    
    def __init__(
        self,
        **kwargs
    ):

        super().__init__(**kwargs)
        model_path = kwargs.pop("model_path", "")
        if model_path:
            if not os.path.exists(model_path):
                # user/repo/filename:revision
                repo_id, filename = model_path.rsplit("/", 1)
                filename, revision = filename.split(":")
                model_path = hf_hub_download(repo_id=repo_id,
                                             filename=filename,
                                             cache_dir=None,
                                             revision=revision,
                                             force_download=False)
            self.model_path = model_path
            pt = torch.load(model_path, weights_only=False)
            self.label2id = pt.get("label2id", self.label2id)
            self.id2label = pt.get("id2label", self.id2label)
            self.transforms = pt["config"].get("data", {}).get("val", {}).get("augment", {})
            self.task = pt["config"].get("model", {}).get("task", None)
            self.num_classes = pt["config"].get("model", {}).get("num_classes", None)
            threshold = 0
            if pt["config"].get('hyp', {}).get('loss', {}).get('bce', [False])[0]:
                threshold = pt["config"]['hyp']['loss']['bce'][1]
                if isinstance(threshold, (int, float)):
                    threshold = [threshold] * self.num_classes
                assert len(threshold) == self.num_classes and isinstance(threshold, list), "threshold must be a list of length num_classes"
            self.threshold = threshold
            self.timm_model = pt["config"].get("model", {}).get("name").split("-")[1]
    
    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path, **kwargs):
        config_dict, kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        for key in list(kwargs.keys()):
            if key in config_dict:
                value = kwargs.pop(key)
                if value is not None:
                    config_dict[key] = value

        return config_dict, kwargs
    
class DoraemonProcessor(ProcessorMixin):
    attributes = []
    config_class = DoraemonConfig
    
    def __init__(self, config: Optional[DoraemonConfig] = None, **kwargs):
        super().__init__()
        self.config = config
        self.transforms = create_AugTransforms(config.transforms)
        self.threshold = np.array(config.threshold)

    def __call__(
        self, 
        image,
        return_tensors: Optional[str] = "pt",
        **kwargs
    ) -> BatchFeature:
            
        image_tensors = self.preprocess(image)
        
        return BatchFeature(
            data={
                "pixel_values": image_tensors,
            },
            tensor_type=return_tensors
        )
    
    def preprocess(self, images, *args, **kwargs):
        if not isinstance(images, list):
            images = [images]
        
        for idx, im in enumerate(images):
            images[idx] = self.transforms(im)
        return torch.stack(images, dim=0)
    
    def postprocess(self, probs):
        batch_size = probs.shape[0]
        results = []

        if self.config.threshold != 0:
            above_threshold = probs > self.config.threshold
            
            for b in range(batch_size):
                result = {self.config.id2label[i]: 0 for i in range(len(self.config.id2label))}
                for i in np.where(above_threshold[b])[0]:
                    result[self.config.id2label[i]] = 1
                results.append(result)
        else:
            indices = np.argmax(probs, axis=1)
            for b in range(batch_size):
                idx = indices[b]
                results.append({self.config.id2label[idx]: float(probs[b, idx])})
        
        return results

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config: DoraemonConfig = kwargs.get("config", None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)        
        return cls(config=config)

class DoraemonClassifier(PreTrainedModel):
    config_class = DoraemonConfig
    
    def __init__(self, config: Optional[DoraemonConfig] = None, **kwargs):
        super().__init__(config)
        self.model = create_model(config.timm_model, pretrained=False, num_classes=config.num_classes)
    
    def forward(self, pixel_values):
        with torch.inference_mode():
            output = self.model(pixel_values)
            if self.config.threshold != 0:
                output = torch.sigmoid(output)
            else:
                output = F.softmax(output, dim=1)

        return output.cpu().numpy()
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = kwargs.get("config", None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        wrapper = cls(config=config)
        
        model_path = config.model_path
        
        pt = torch.load(model_path, weights_only=False)
        wrapper.model.load_state_dict(pt['ema'].state_dict() if pt.get("ema", False) else pt['model'])
        wrapper.model.eval()

        return wrapper
if __name__ == "__main__":
    from transformers import AutoModel, AutoProcessor
    import requests
    from io import BytesIO
    from PIL import Image
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser(description='Doraemon Model Inference Deploy With Huggingface')
        parser.add_argument('--pretrained_model_name_or_path', type=str, default='./', 
                            help='pretrained model name or path')
        parser.add_argument('--revision', type=str, default=None, 
                        help='model revision')
        parser.add_argument('--model_path', type=str, default=None, 
                            help='If given, it will be override the model_path in config.json')
        args = parser.parse_args()

        return args

    args = parse_args()
    print(args)
    model = AutoModel.from_pretrained(args.pretrained_model_name_or_path, 
                                      revision=args.revision,
                                      model_path=args.model_path,
                                      trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.pretrained_model_name_or_path, 
                                         revision=args.revision,
                                         model_path=args.model_path,
                                         trust_remote_code=True)

    urls = [
        "https://github.com/ultralytics/ultralytics/blob/main/ultralytics/assets/bus.jpg",
        "https://github.com/ultralytics/ultralytics/blob/main/ultralytics/assets/zidane.jpg",
    ]

    images = []
    for url in urls:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        images.append(image)

    batch_inputs = processor(images, return_tensors="pt")

    probs = model(batch_inputs["pixel_values"])
    print("Probs:\n", probs)
    output = processor.postprocess(probs)
    print("Tagging:")
    for r in output:
        print(r)