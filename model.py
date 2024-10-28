import torch
from dataloader import BatchedImages
from swin_transformer import SwinTransformer
from constants import IMG_SIZE

class DepthAndUncertaintyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
         # args from https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py
        self.swint = SwinTransformer(
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False

        )
        self.layer_one = torch.nn.Linear(768,2)
    
    def forward(self, x: BatchedImages):
        x = self.swint(x.rgb)
        x = x[-1] # torch.Size([16, 768, 20, 15])
        x= torch.nn.functional.interpolate(x, mode="bilinear", size=IMG_SIZE)
        x = x.permute(0,2,3,1)
        x = self.layer_one(x)
        return x
