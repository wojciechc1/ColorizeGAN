import torch
from src.data.dataset import GrayscaleColorDataset, get_dataloader


def test_dataset_shapes():
    dataset = GrayscaleColorDataset(train=True)
    gray, color = dataset[0]

    assert isinstance(gray, torch.Tensor)
    assert isinstance(color, torch.Tensor)
    assert gray.shape[0] == 1  # grayscale channel
    assert color.shape[0] == 3  # RGB channels


def test_dataloader_shapes():
    train_loader, val_loader = get_dataloader(
        batch_size=4, train_samples=8, val_samples=2
    )
    for gray, color in train_loader:
        assert gray.shape[1] == 1
        assert color.shape[1] == 3
        break
