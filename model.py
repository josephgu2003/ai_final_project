import torch
from dataloader import BatchedImages
from swin_transformer import SwinTransformer
from constants import IMG_SIZE

import torch.nn.functional as F

class DepthAndUncertaintyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
         # args from https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py
        self.backbone = SwinTransformer(
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False

        )
        self.layer_one = torch.nn.Linear(768,384)
        self.layer_two = torch.nn.Linear(384,192)
        self.layer_three = torch.nn.Linear(192,2)
    
    def apply_linear(self, linear, x):
        return linear(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    def forward(self, x: BatchedImages):
        out = self.backbone(x.rgb)
        x = out[-1] # torch.Size([16, 768, 20, 15])

        x = F.relu(self.apply_linear(self.layer_one, x))

        x= torch.nn.functional.interpolate(x, mode="bilinear", size=(30, 40))

        x += out[-2]

        x = F.relu(self.apply_linear(self.layer_two, x))

        x= torch.nn.functional.interpolate(x, mode="bilinear", size=(60, 80))

        x += out[-3]

        x = self.apply_linear(self.layer_three, x)

        x= torch.nn.functional.interpolate(x, mode="bilinear", size=IMG_SIZE)
    
        return x.permute(0, 2, 3, 1)
