from utils import get_dataloader, imshow, plot_losses
import torch
from model import UNet, Discriminator
from train import train
import torch.nn as nn

# hyperparameters
batch_size = 32
num_epochs = 50

train_samples = 1000  # max 60_000
val_samples = 100  # max 10_000

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data:
train_loader, val_loader = get_dataloader(
    batch_size=batch_size, train_samples=train_samples, val_samples=val_samples
)

print(len(train_loader))

# models
g = UNet().to(device)
d = Discriminator().to(device)
#g.load_state_dict(torch.load("generator.pth", map_location=device))
#d.load_state_dict(torch.load("discriminator.pth", map_location=device))


criterion_bce = nn.BCEWithLogitsLoss()
criterion_l1 = nn.L1Loss()

g_opt = torch.optim.Adam(g.parameters(), lr=0.00001, betas=(0.7, 0.999))
d_opt = torch.optim.Adam(d.parameters(), lr=0.00001, betas=(0.7, 0.999))

from torch.optim.lr_scheduler import StepLR


g_scheduler = StepLR(g_opt, step_size=4, gamma=1)
d_scheduler = StepLR(d_opt, step_size=4, gamma=1)

all_d_losses = []
all_g_losses = []

for epoch in range(num_epochs):
    torch.cuda.empty_cache()
    print("Epoch {}/{}".format(epoch + 1, num_epochs))
    d_loss, g_loss = train(train_loader, g, d, criterion_l1, criterion_bce, d_opt, g_opt, device)

    all_d_losses.append(d_loss)
    all_g_losses.append(g_loss)

    g_scheduler.step()
    d_scheduler.step()

    for gray_images, color_images in val_loader:
        gray_images = gray_images.to(device)
        color_images = color_images.to(device)
        break  # tylko pierwszy batch

    with torch.no_grad():
        fake_imm = g(gray_images)
        imshow(color_images[0], gray_images[0], fake_imm[0])

plot_losses(all_d_losses, all_g_losses)
# zapis wag generatora
torch.save(g.state_dict(), "generator.pth")

# zapis wag dyskryminatora
torch.save(d.state_dict(), "discriminator.pth")