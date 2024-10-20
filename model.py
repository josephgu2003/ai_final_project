import torch
from SwinTransformer.models.swin_transformer_v2 import SwinTransformerV2
from constants import IMG_SIZE

class DepthAndUncertaintyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        print("Entering forward...")
        swint = SwinTransformerV2(
            img_size = IMG_SIZE,
            depth

        )
        x = swint(x)
        return x




def signature(rgb: torch.tensor) -> torch.tensor:
    """
    Intake the image, output the depthmap and uncertainty
    :param rgb: b 3 h w
    :returns : b 2 h w
    """
    res = DepthAndUncertaintyModel()(rgb)
    print(res)
    return res

# ViT 
# SwinT !!

if __name__ == "__main__":
    signature(torch.tensor([
        [[[5.0]*IMG_SIZE]*IMG_SIZE,[[4.0]*IMG_SIZE]*IMG_SIZE,[[3.0]*IMG_SIZE]*IMG_SIZE]
    ]))