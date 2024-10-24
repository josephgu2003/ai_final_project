import torch


from constants import IMG_SIZE
from swin_transformer import SwinTransformer
import numpy as np


x = SwinTransformer()(torch.tensor([
        [[[5.0]*IMG_SIZE]*IMG_SIZE,[[4.0]*IMG_SIZE]*IMG_SIZE,[[3.0]*IMG_SIZE]*IMG_SIZE]
    ]))