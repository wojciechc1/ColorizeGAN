import torch
from src.data.dataset import GrayscaleColorDataset, get_dataloader


def test_dataset_shapes(monkeypatch):
    # Mock CIFAR10
    class DummyCIFAR:
        def __init__(self, *args, **kwargs):
            # 5 imgs RGB 3x32x32
            self.data = torch.randn(5, 3, 32, 32)
            self.targets = [0, 1, 2, 3, 4]

        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]

        def __len__(self):
            return len(self.data)

    monkeypatch.setattr("torchvision.datasets.CIFAR10", DummyCIFAR)

    dataset = GrayscaleColorDataset(train=True)
    gray, color = dataset[0]

    assert gray.shape == (1, 32, 32)  # grayscale
    assert color.shape == (3, 32, 32)  # color


def test_dataloader_shapes(monkeypatch):
    # Mock GrayscaleColorDataset
    class DummyDataset:
        def __init__(self, *args, **kwargs):
            self.data = torch.randn(10, 3, 32, 32)

        def __getitem__(self, idx):
            color = self.data[idx]
            gray = color.mean(dim=0, keepdim=True)  # grayscale [1,H,W]
            return gray, color

        def __len__(self):
            return len(self.data)

    monkeypatch.setattr("src.data.dataset.GrayscaleColorDataset", DummyDataset)

    train_loader, val_loader = get_dataloader(
        batch_size=4, train_samples=8, val_samples=2
    )

    for gray, color in train_loader:
        assert gray.shape[1] == 1
        assert color.shape[1] == 3
        break
