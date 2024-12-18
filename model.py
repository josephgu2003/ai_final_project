import torch
from dataloader import BatchedImages
from layers import AttnDecoderBlock, ConvDecoderBlock, MCDropout
from swin_transformer import SwinTransformer, SwinTransformerBlock
from constants import IMG_SIZE

import torch.nn.functional as F

class DepthAndUncertaintyModel(torch.nn.Module):
    def __init__(self, args):
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
        
        self.attn_one = AttnDecoderBlock(768, 384, 30, 40)
        self.conv_two = ConvDecoderBlock(384, 192, 60, 80)
        self.conv_three = ConvDecoderBlock(192, 96, 120, 160)
        self.conv_four = ConvDecoderBlock(96, 2 if args.use_aleatoric else 1, IMG_SIZE[0], IMG_SIZE[1], act=torch.nn.Identity)
        
        self.mc_dropout = MCDropout(p=0.5) if args.use_epistemic else torch.nn.Identity()
        
    def process(self, x: BatchedImages):
        out = self.backbone(x.rgb)
        x = out[-1] # torch.Size([16, 768, 20, 15])
        x = self.attn_one(x)
        x = self.mc_dropout(x)
        x = self.conv_two(x, out[-2])
        x = self.mc_dropout(x)
        x = self.conv_three(x, out[-3])
        x = self.mc_dropout(x)
        x = self.conv_four(x, out[-4])
    
        return x.permute(0, 2, 3, 1)
    
    def forward(self, x: BatchedImages, samples: int = 1):
        preds = torch.stack(list([self.process(x) for i in range(samples)]))
        preds_var = torch.var(preds, dim=0)
        preds = torch.mean(preds, dim=0)
        return preds, preds_var
