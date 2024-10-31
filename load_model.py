import torch
from model import DepthAndUncertaintyModel
import os

def load_model(path, model):
    checkpoint = torch.load(path, map_location='cpu')
    errors = model.load_state_dict(checkpoint['state_dict'], False)

    return model, errors

def build_model(args, device):
    model = DepthAndUncertaintyModel()
    model, errors = load_model(os.path.join(args.model_dir, 'upernet_swin_tiny_patch4_window7_512x512.pth'), model)
    model = model.to(device)
    return model, errors
