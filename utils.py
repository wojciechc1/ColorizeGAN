import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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



def imshow(gray_image, color_image):
    """
        Displays a grayscale and corresponding color image side by side using matplotlib.

        Args:
            gray_image (torch.Tensor): A tensor representing the grayscale image with shape [1, H, W].
            color_image (torch.Tensor): A tensor representing the color image with shape [3, H, W].

        The function squeezes and permutes the tensors as needed, converts them to NumPy arrays,
        and shows them using matplotlib with appropriate titles and layout.
    """

    gray_image_np = gray_image.squeeze().numpy()
    color_image_np = color_image.permute(1, 2, 0).numpy()
    color_image_np = (color_image_np + 1) / 2  # <- This rescales to [0, 1]

    plt.subplot(1, 2, 1)
    plt.title("Grayscale")
    plt.imshow(gray_image_np, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Color")
    plt.imshow(color_image_np)
    plt.axis('off')

    plt.tight_layout()
    plt.show()