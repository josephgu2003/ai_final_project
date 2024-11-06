import torch
import numpy

from dataloader import BatchedImages
from swin_transformer import SwinTransformer
from constants import IMG_SIZE, DECODER_LAYER_CT

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
        BACKBONE_OUT_DEPTH = 768
        BACKBONE_OUT_SIZE = 16
        self.decoder_layer_depths = np.linspace(BACKBONE_OUT_DEPTH,2, num=DECODER_LAYER_CT+1)
        self.decoder_linear_layers = [torch.nn.Linear(self.decoder_layer_depths[i],self.decoder_layer_depths[i+1]) for i in range(DECODER_LAYER_CT)]
        self.decode_interp_sizes = np.linspace(BACKBONE_OUT_SIZE, IMG_SIZE, DECODER_LAYER_CT)
    
    def forward(self, x: BatchedImages):
        x = self.backbone(x.rgb)
        x = x[-1] # torch.Size([16, 768, 20, 15])
        for i in range(DECODER_LAYER_CT):
            x = x.permute(0,2,3,1)
            x= self.decoder_linear_layers[i](x)
            x = x.permute(0,3,1,2)
            x= torch.nn.functional.interpolate(x, mode="bilinear", size=self.decode_interp_sizes[i])
        return x
