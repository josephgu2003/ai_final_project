import torch

def signature(rgb: torch.tensor) -> torch.tensor:
    """
    Intake the image, output the depthmap and uncertainty
    :param rgb: b 3 h w
    :returns : b 2 h w
    """

# ViT 
# SwinT !!