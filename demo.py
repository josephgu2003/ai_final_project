import sys
import numpy as np
from PIL import Image
import torch
import configargparse
from torchvision.transforms import ToTensor, ToPILImage
from pathlib import Path
from dataloader import BatchedImages
from load_model import load_model
from model import DepthAndUncertaintyModel

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('image_file', type=str)

    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()
args = config_parser(sys.argv[1:])
model, err = load_model(args.checkpoint, DepthAndUncertaintyModel())
if err is not None:
    print(err, file=sys.stderr)
out = model(BatchedImages(ToTensor()(Image.open(args.image_file)).unsqueeze(0), None))
out = out[0].permute(2,0,1).unsqueeze(1).repeat(1,3,1,1)
ToPILImage()(out[0]).save(f"depth_{args.image_file}")
ToPILImage()(out[1]).save(f"uncertainty_{args.image_file}")
