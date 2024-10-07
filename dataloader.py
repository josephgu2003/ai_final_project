from dataclasses import dataclass
import torch


@dataclass 
class BatchedImages:
    rgb: torch.tensor #b 3 h w
    label: torch.tensor #b 1 h w

@dataclass 
class LabeledImage:
    rgb: torch.tensor #3 h w
    label: torch.tensor #1 h w
    filename: str

def collate_fn(data: list[LabeledImage]):
    pass

#https://paperswithcode.com/datasets?task=depth-estimation

# KITTI
# NYUv2
# Matterport