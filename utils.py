import torch
from torchvision import datasets, transforms


class GrayscaleColorDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self.dataset = datasets.CIFAR10(
            root='./data',
            train=train,
            download=True,
            transform=transforms.ToTensor()
        )
        self.to_grayscale = transforms.Grayscale(num_output_channels=1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        color_img, _ = self.dataset[idx]  # [3, 32, 32]
        gray_img = self.to_grayscale(color_img)  # [1, 32, 32]
        return gray_img, color_img

