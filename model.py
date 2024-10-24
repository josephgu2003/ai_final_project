import torch
from swin_transformer import SwinTransformer
from constants import IMG_SIZE

class DepthAndUncertaintyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        print("Entering forward...")
        # args from https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py
        swint = SwinTransformer(
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False

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