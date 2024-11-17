import torch
from dataloader import BatchedImages
from swin_transformer import SwinTransformer, SwinTransformerBlock
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
        self.conv_one = torch.nn.Conv2d(768, 384, kernel_size=(3,3),padding=1)
        # window size 20 means it is like a full attention layer
        self.attn = SwinTransformerBlock(dim=384, num_heads=6, window_size=20, attn_drop=0.3) # will accept a [384, 20, 15] input after reshaping
        self.attn.H = 15
        self.attn.W = 20

        self.conv_two = torch.nn.Conv2d(384, 192, kernel_size=(3,3),padding=1)
        self.conv_three = torch.nn.Conv2d(192, 2, kernel_size=(3,3),padding=1)
        
    def apply_linear(self, linear, x):
        return linear(x)
    
    def forward(self, x: BatchedImages):
        out = self.backbone(x.rgb)
        x = out[-1] # torch.Size([16, 768, 20, 15])

        x = F.relu(self.apply_linear(self.conv_one, x))

        x = self.attn(x.flatten(-2).permute(0, 2, 1), mask_matrix=None).permute(0, 2, 1).reshape(x.shape)

        x= torch.nn.functional.interpolate(x, mode="bilinear", size=(30, 40))

        x += out[-2]

        x = F.relu(self.apply_linear(self.conv_two, x))

        x= torch.nn.functional.interpolate(x, mode="bilinear", size=(60, 80))

        x += out[-3]

        x = self.apply_linear(self.conv_three, x)

        x= torch.nn.functional.interpolate(x, mode="bilinear", size=IMG_SIZE)
    
        return x.permute(0, 2, 3, 1)
