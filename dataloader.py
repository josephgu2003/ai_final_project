from dataclasses import dataclass
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np

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
    def __init__(self, mat_file_path: str, transform=None):
        self.data = h5py.File(mat_file_path, 'r')

        self.rgb_images = self.data['images']
        self.labels = self.data['depths']

        self.transform = transform

    def __len__(self):
        return self.rgb_images.shape[0]

    def __getitem__(self, idx):
        rgb_image = np.array(self.rgb_images[idx])
        label = np.array(self.labels[idx])

        rgb_image = Image.fromarray(
            np.uint8(rgb_image.transpose(1, 2, 0)))
        label = Image.fromarray(np.float32(label))

        rgb_tensor = ToTensor()(rgb_image)  # Shape: (3, H, W)
        label_tensor = ToTensor()(label)  # Shape: (1, H, W)

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
