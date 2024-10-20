from constants import IMG_SIZE
from mmseg.models.backbones import SwinTransformer
import numpy as np
import torch

x = SwinTransformer()(torch.tensor([
        [[[5.0]*IMG_SIZE]*IMG_SIZE,[[4.0]*IMG_SIZE]*IMG_SIZE,[[3.0]*IMG_SIZE]*IMG_SIZE]
    ]))