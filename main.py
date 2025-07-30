from utils import get_dataloader, imshow
import torch

# hyperparameters
batch_size = 10000
num_epochs = 1

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# load data:
train_loader, val_loader = get_dataloader(batch_size=batch_size)



for epoch in range(num_epochs):
    for gray_images, color_images in train_loader:
        imshow(gray_images[0], color_images[0])
