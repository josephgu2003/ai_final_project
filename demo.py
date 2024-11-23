import sys

from PIL import Image
import configargparse
import torchvision
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

args = config_parser(sys.argv)
model, err = load_model(args.checkpoint, DepthAndUncertaintyModel())
if err is not None:
    print(err, file=sys.stderr)
out = model(BatchedImages(torchvision.transforms.ToTensor()(Image.open(args.file)), None))
torchvision.transforms.toPILImage()(out.rgb[:,:,:,0]).save(f"depth_{args.file}")
torchvision.transforms.toPILImage()(out.rgb[:,:,:,1]).save(f"uncertainty{args.file}")
