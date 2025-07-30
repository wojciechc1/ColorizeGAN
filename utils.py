import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class GrayscaleColorDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.dataset = datasets.CIFAR10(
            root='./data',
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
        )
        self.to_grayscale = transforms.Grayscale(num_output_channels=1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        color_img, _ = self.dataset[idx]  # [3, 32, 32]
        gray_img = self.to_grayscale(color_img)  # [1, 32, 32]
        return gray_img, color_img


def get_dataloader(batch_size=32):
    """
        Returns DataLoaders for the CIFAR-10 dataset with paired grayscale and color images.

        The dataset consists of grayscale inputs and corresponding color targets, useful for image colorization tasks.

        Args:
            batch_size (int): Number of samples per batch. Default is 32.

        Returns:
            Tuple[DataLoader, DataLoader]:
                - train_loader: DataLoader for training data.
                - val_loader: DataLoader for validation data.
    """
    train_loader = DataLoader(
        GrayscaleColorDataset(train=True),
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        GrayscaleColorDataset(train=False),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader