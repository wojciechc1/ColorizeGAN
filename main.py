from utils import get_dataloader
import torch

# hyperparameters
batch_size = 64
num_epochs = 1

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# load data:
train_loader, val_loader = get_dataloader(batch_size=batch_size)



for epoch in range(num_epochs):
    for img in train_loader:
        print(img)