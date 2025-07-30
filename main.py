from utils import get_dataloader, imshow, plot_losses
import torch
from model import UNet, Discriminator
from train import train

# hyperparameters
batch_size = 32
num_epochs = 100
max_samples = 100

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# load data:
train_loader, val_loader = get_dataloader(batch_size=batch_size, max_samples=max_samples)
print(len(train_loader))

# models
g = UNet()
d = Discriminator()

import torch.nn as nn

criterion_bce = nn.BCEWithLogitsLoss()
criterion_l1 = nn.L1Loss()

g_opt = torch.optim.Adam(g.parameters(), lr=0.0003, betas=(0.5, 0.999))
d_opt = torch.optim.Adam(d.parameters(), lr=0.0001, betas=(0.5, 0.999))

# metrics
all_d_losses = []
all_g_losses = []

for epoch in range(num_epochs):
    d_loss, g_loss = train(train_loader, g, d, criterion_l1, criterion_bce, d_opt, g_opt)

    all_d_losses.append(d_loss)
    all_g_losses.append(g_loss)

    batch = next(iter(train_loader))
    gray_images, color_images = batch

    with torch.no_grad():
        fake_imm = g(gray_images)
        imshow(color_images[0], gray_images[0], fake_imm[0])


plot_losses(all_d_losses, all_g_losses)