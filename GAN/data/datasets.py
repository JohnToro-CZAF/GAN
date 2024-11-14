# GAN/data/datasets.py

import os
from PIL import Image
from torch.utils.data import Dataset


class GANDataset(Dataset):
    """Custom Dataset for unpaired image-to-image translation."""
    def __init__(self, root_dir_A, root_dir_B, transform=None):
        """
        Args:
            root_dir_A (string): Directory with all the images from domain A.
            root_dir_B (string): Directory with all the images from domain B.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir_A = root_dir_A
        self.root_dir_B = root_dir_B
        self.transform = transform

        self.images_A = sorted(os.listdir(root_dir_A))
        self.images_B = sorted(os.listdir(root_dir_B))

    def __len__(self):
        return max(len(self.images_A), len(self.images_B))

    def __getitem__(self, idx):
        img_A_path = os.path.join(self.root_dir_A, self.images_A[idx % len(self.images_A)])
        img_B_path = os.path.join(self.root_dir_B, self.images_B[idx % len(self.images_B)])

        image_A = Image.open(img_A_path).convert('RGB')
        image_B = Image.open(img_B_path).convert('RGB')

        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)

        return {'A': image_A, 'B': image_B}

class PairedGANDataset(Dataset):
    """Dataset for paired image-to-image translation (e.g., Pix2Pix)."""
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the paired images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = sorted(os.listdir(root_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        width, height = image.size
        image_A = image.crop((0, 0, width // 2, height))
        image_B = image.crop((width // 2, 0, width, height))

        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)

        return {'A': image_A, 'B': image_B}
