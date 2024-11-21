import torch 
import torch.nn as nn 
import torch.nn.functional as F

from swin_transformer import SwinTransformerBlock

# dropout that is used at training and inference time
class MCDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p 
        
    def forward(self, x):
        return F.dropout(x, p = self.p, training = True)
        
class AttnDecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, out_h, out_w):
        super().__init__()
        self.attn = SwinTransformerBlock(dim=in_dim, num_heads=6, window_size=20, attn_drop=0.3) # will accept a [384, 20, 15] input after reshaping
        
        self.attn.H = 15
        self.attn.W = 20
        
        self.out_proj = nn.Linear(in_dim, out_dim)
        self.out_dim = out_dim
        
        self.out_h = out_h 
        self.out_w = out_w
        
    def forward(self, x):
        b, d, h, w = x.shape
        x = self.attn(x.flatten(-2).permute(0, 2, 1), mask_matrix=None)
        x = F.relu(self.out_proj(x))
        x = x.permute(0, 2, 1)
        x = x.reshape(b, self.out_dim, h, w)

        return torch.nn.functional.interpolate(x, mode="bilinear", size=(self.out_h, self.out_w))
        
class ConvDecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, out_h, out_w, act=nn.ReLU):
        super().__init__()
        
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), padding=1)
        self.out_h = out_h 
        self.out_w = out_w 
        
        self.act = act()
        
    def forward(self, x, res=None):
        if res is not None:
            x = x + res

        x = self.conv(x)
        x = self.act(x)
        return F.interpolate(x, mode="bilinear", size=(self.out_h, self.out_w))
        
    
