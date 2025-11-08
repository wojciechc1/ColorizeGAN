import random
from typing import Tuple
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from src.config import CONFIG


class GrayscaleColorDataset(torch.utils.data.Dataset):
    """CIFAR-10 dataset with paired grayscale and color images."""

    def __init__(self, train: bool = True):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.dataset = datasets.CIFAR10(
            root=CONFIG["dataset_path"],
            train=train,
            download=CONFIG["download"],
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )
        self.to_grayscale = transforms.Grayscale(num_output_channels=1)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        color_img, _ = self.dataset[idx]  # [3, H, W]
        gray_img = self.to_grayscale(color_img)  # [1, H, W]
        return gray_img, color_img


def get_dataloader(
    batch_size: int, train_samples: int, val_samples: int
) -> Tuple[DataLoader, DataLoader]:
    """Returns train and validation DataLoaders with subsets."""
    random.seed(42)

    full_train_dataset = GrayscaleColorDataset(train=True)
    full_val_dataset = GrayscaleColorDataset(train=False)

    train_indices = random.sample(range(len(full_train_dataset)), train_samples)
    val_indices = random.sample(range(len(full_val_dataset)), val_samples)

    train_loader = DataLoader(
        Subset(full_train_dataset, train_indices), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        Subset(full_val_dataset, val_indices), batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader
