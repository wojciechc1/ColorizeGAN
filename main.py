from utils import GrayscaleColorDataset
from torch.utils.data import DataLoader

# Użycie:
batch_size = 64

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