"""
A collection of functions which:
    Returns the image and label flipped horizontally with probability p
    :param image: 3 x H x W
    :param label: 1 x H x W
    :return: (3 x H x W, 1 x H x W)

    Also contains a compose function for composing them
"""


import torch
from torchvision.transforms.v2 import RandomHorizontalFlip, RandomChannelPermutation
from PIL.Image import Image
from typing import Callable
import random

def horizontal_flip(image: Image, label: Image, p=0.5) -> tuple[torch.Tensor,torch.Tensor]:
    random_horizontal_flip = RandomHorizontalFlip(p)
    return random_horizontal_flip(image, label)

def channel_shift(image: Image, label: Image, p):
    """Shifts color channel randomly"""
    random.choice(range(3))
    r = RandomChannelPermutation()
    return r(image, label)

def compose(img: Image, label: Image, transforms: Callable[[Image, Image],tuple[torch.Tensor,torch.Tensor]]):
    img = img
    label = label
    for transform in transforms:
        img, label = transform(img, label)
    return img,label