"""
A collection of functions which:
    Returns the image and label flipped horizontally with probability p
    :param image: 3 x H x W
    :param label: 1 x H x W
    :return: (3 x H x W, 1 x H x W)

    Also contains a compose function for composing them
"""
import torch
import torchvision.transforms.v2 as T
from PIL.Image import Image
from typing import Callable
from random import random

def horizontal_flip(p):
    res = T.RandomHorizontalFlip(p)
    return res, res

def channel_shift(p):
    return (T.RandomChannelPermutation() if random() < p else T.Identity()), T.Identity()

def cutout(p):
    res = T.Compose(T.PILToTensor(), T.RandomErasing(p), T.ToPILImage())
    return res, res

def affine(max_angle, max_translate_pct, fill_color):
    res = T.RandomAffine(degrees=max_angle, translate=(max_translate_pct,max_translate_pct), fill=fill_color)
    return res, res

def grayscale(p):
    return T.RandomGrayscale(p), T.Identity()

def compose(transforms: Callable[[Image, Image],tuple[torch.Tensor,torch.Tensor]]):
    img_t, label_t = map(T.Compose, zip(transforms))
    def res(img, label):
        return img_t(img), label_t(label)
    return res

