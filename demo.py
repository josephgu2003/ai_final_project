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
import matplotlib.pyplot as plt

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
out = out[0].permute(2,0,1)

def normalize(im):
    return (im-torch.min(im))/(torch.max(im)-torch.min(im))

depth = normalize(out[0])
uncertainty = normalize(out[1])

def saveImage(img, filename):
    img = plt.get_cmap('plasma')(img.detach().numpy())[:,:,:3] * 255
    Image.fromarray(torch.tensor(img, dtype=torch.uint8).cpu().numpy()).save(filename)

saveImage(depth, f"depth_{args.image_file}")
saveImage(uncertainty, f"uncertainty_{args.image_file}")
