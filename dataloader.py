from dataclasses import dataclass
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, RandomHorizontalFlip, Compose
from PIL import Image
import numpy as np
import scipy.io as io

@dataclass
class BatchedImages:
    rgb: torch.tensor #b 3 h w
    label: torch.tensor #b 1 h w

@dataclass
class LabeledImage:
    rgb: torch.tensor #3 h w
    label: torch.tensor #1 h w
    filename: str

class NYUv2Dataset(Dataset):
    def __init__(self, mat_file_path: str, splits_path: str, mode: str = 'train'):
        self.data = h5py.File(mat_file_path, 'r')
        
        if mode == 'train':
            indices = io.loadmat(splits_path)['trainNdxs']
            self.transform = Compose([RandomHorizontalFlip(p=0.5), ToTensor()])
        elif mode == 'test':
            indices = io.loadmat(splits_path)['testNdxs']
            self.transform = Compose([ToTensor()])
        else: 
            raise ValueError()
        indices = indices.flatten() - 1

        self.rgb_images = np.array(self.data['images'])[indices]
        self.labels = np.array(self.data['depths'])[indices]
        

    def __len__(self):
        return self.rgb_images.shape[0]

    def __getitem__(self, idx):
        rgb_image = self.rgb_images[idx]
        label = self.labels[idx]

        rgb_image = Image.fromarray(
            np.uint8(rgb_image.transpose(1, 2, 0)))
        label = Image.fromarray(np.float32(label))

        rgb_tensor = self.transform(rgb_image)  # Shape: (3, H, W)
        label_tensor = self.transform(label)  # Shape: (1, H, W)

        return LabeledImage(
            rgb=rgb_tensor,
            label=label_tensor,
            filename=f"image_{idx}.png"
        )

def collate_fn(labeled_imgs: list[LabeledImage]):
    batched_imgs = BatchedImages(
        rgb=torch.stack([labeled_img.rgb for labeled_img in labeled_imgs]),
        label=torch.stack([labeled_img.label for labeled_img in labeled_imgs])
    )
    return batched_imgs
