import torch
from .SwinTransformer.models.swin_transformer_v2 import SwinTransformerV2

class DepthAndUncertaintyModel(torch.nn.Module):
    def __init__():
        super().__init__()
    
    def forward(self, x):
        swint = SwinTransformerV2()
        return swint(x)



def signature(rgb: torch.tensor) -> torch.tensor:
    """
    Intake the image, output the depthmap and uncertainty
    :param rgb: b 3 h w
    :returns : b 2 h w
    """
    return DepthAndUncertaintyModel()(rgb)

# ViT 
# SwinT !!