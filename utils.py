import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random

import matplotlib.pyplot as plt


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


def get_dataloader(batch_size=32, train_samples=1000, val_samples=100):
    """
    Returns DataLoaders for the CIFAR-10 dataset with paired grayscale and color images.

    The dataset consists of grayscale inputs and corresponding color targets,
    useful for image colorization tasks.

    Args:
        batch_size (int): Number of samples per batch. Default is 32.
        train_samples (int): Maximum number of samples to load for test.
        val_samples (int): Maximum number of samples to load for validation.

    Returns:
        Tuple[DataLoader, DataLoader]:
            - train_loader: DataLoader for training data.
            - val_loader: DataLoader for validation data.
    """
    full_train_dataset = GrayscaleColorDataset(train=True)
    full_val_dataset = GrayscaleColorDataset(train=False)

    train_indices = random.sample(range(len(full_train_dataset)), train_samples)
    val_indices = random.sample(range(len(full_val_dataset)), val_samples)

    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_val_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader



def imshow(real_img, gray_img, fake_img):
    """
    Displays grayscale, real color, and generated color images side by side using matplotlib.

    Args:
        real_img (torch.Tensor): Real color image tensor with shape [3, H, W], values in [-1, 1].
        gray_img (torch.Tensor): Grayscale image tensor with shape [1, H, W].
        fake_img (torch.Tensor): Generated color image tensor with shape [3, H, W], values in [-1, 1].
    """
    import matplotlib.pyplot as plt

    # Przygotowanie obrazów do wyświetlenia
    gray_np = gray_img.squeeze().cpu().numpy()
    real_np = (real_img.permute(1, 2, 0).cpu().numpy() + 1) / 2  # skalowanie do [0,1]
    fake_np = (fake_img.permute(1, 2, 0).cpu().numpy() + 1) / 2  # skalowanie do [0,1]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Real Color")
    plt.imshow(real_np)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Grayscale")
    plt.imshow(gray_np, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Generated Color")
    plt.imshow(fake_np)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_losses(all_d_losses, all_g_losses):
    """
    Plots the training losses of the discriminator and generator.

    Args:
        all_d_losses (List[float]): List of discriminator losses.
        all_g_losses (List[float]): List of generator losses.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(all_d_losses, label="Discriminator Loss")
    plt.plot(all_g_losses, label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()