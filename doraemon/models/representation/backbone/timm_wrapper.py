import timm
import torch
import torch.nn as nn

class TimmWrapper(nn.Module):
    """A wrapper for timm models that handles different model architectures uniformly."""
    
    def __init__(self, 
                 model_name: str,
                 feat_dim: int,
                 image_size: int,
                 pretrained: bool = True,
                 **kwargs):
        super().__init__()
        
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # classification head -> nn.Identity()
            global_pool='',  # global pooling -> nn.Identity()
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, image_size, image_size)
            output = self.model(dummy_input)
            
            if isinstance(output, tuple):
                output = output[0]
            
            if len(output.shape) == 4:  # CNN output: [B, C, H, W]
                _, channels, h, w = output.shape
                flatten_dim = channels * h * w
                self.output_layer = nn.Sequential(
                    nn.BatchNorm2d(channels),
                    nn.Flatten(1),
                    nn.Linear(flatten_dim, feat_dim),
                    nn.BatchNorm1d(feat_dim)
                )
            elif len(output.shape) == 3:  # Transformer output: [B, N, C]
                _, tokens, channels = output.shape
                flatten_dim = tokens * channels
                self.output_layer = nn.Sequential(
                    nn.LayerNorm(channels),
                    nn.Flatten(1),
                    nn.Linear(flatten_dim, feat_dim),
                    nn.BatchNorm1d(feat_dim)
                )
            else:
                raise ValueError(f"Unexpected output shape: {output.shape}")

    def forward(self, x):
        x = self.model(x)
        x = self.output_layer(x)
        return x