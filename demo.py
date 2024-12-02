from argparse import Namespace
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
import matplotlib.image as mpimg
import os

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('image_file', type=str)

    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()
args = config_parser(sys.argv[1:])

import yaml

with open(os.path.join(os.path.dirname(args.checkpoint), 'saved_config.yaml')) as stream:
    try:
        model_args = Namespace(**yaml.safe_load(stream))
        print(model_args)
    except yaml.YAMLError as exc:
        print(exc)
        
model, err = load_model(args.checkpoint, DepthAndUncertaintyModel(model_args))

model.eval()

if err is not None:
    print(err, file=sys.stderr)

def saveImage(img, filename, cmap='plasma', title="None"):
    img = plt.get_cmap(cmap)(img.detach().numpy())[:,:,:3] * 255
    Image.fromarray(torch.tensor(img, dtype=torch.uint8).cpu().numpy()).save(filename)
    plt.imshow(torch.tensor(img, dtype=torch.uint8).cpu().numpy().astype('uint8'))
    plt.axis('off') 
    plt.title(title)
    plt.show()

img = mpimg.imread(args.image_file)
plt.imshow(img)
plt.title('input')
plt.axis('off')  # Optional: Turn off axis labels and ticks
plt.show()

img = ToTensor()(Image.open(args.image_file)).unsqueeze(0)
out, var = model(BatchedImages(img, None), 8)
out = out[0].permute(2,0,1)
var = var[0].permute(2,0,1)

def normalize(im):
    return (im-torch.min(im))/(torch.max(im)-torch.min(im))

depth = normalize(out[0])
saveImage(depth, f"depth_{args.image_file}", title='depth')

if args.use_aleatoric:
    uncertainty = normalize(torch.exp(out[1]))
    saveImage(uncertainty, f"uncertainty_{args.image_file}", cmap='hot', title='Aleatoric Uncertainty')
    
saveImage(normalize(var[0]), f"dropout_uncertainty_{args.image_file}", cmap='hot', title='Epistemic Uncertainty')
