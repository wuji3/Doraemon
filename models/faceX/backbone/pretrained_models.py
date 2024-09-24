from torchvision.models.convnext import ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights, ConvNeXt_Large_Weights

_prefix = "torchvision"

PreTrainedModels = {
    _prefix + "-convnext_tiny": ConvNeXt_Tiny_Weights,
    _prefix + "-convnext_small": ConvNeXt_Small_Weights,
    _prefix + "-convnext_base": ConvNeXt_Base_Weights,
    _prefix + "-convnext_large": ConvNeXt_Large_Weights
}