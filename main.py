from utils import get_dataloader, imshow, plot_losses
import torch
from model import UNet, Discriminator
from train import train
import torch.nn as nn

# hyperparameters
batch_size = 32
num_epochs = 50

train_samples = 200  # max 60_000
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
#g.load_state_dict(torch.load("saved/generator.pth", map_location=device))
#d.load_state_dict(torch.load("saved/discriminator.pth", map_location=device))


criterion_bce = nn.BCEWithLogitsLoss()
criterion_l1 = nn.L1Loss()

g_opt = torch.optim.Adam(g.parameters(), lr=0.0001, betas=(0.5, 0.999))
d_opt = torch.optim.Adam(d.parameters(), lr=0.0001, betas=(0.4, 0.999))

from torch.optim.lr_scheduler import StepLR


class AdaptiveLRScheduler:
    """
    Dostosowuje LR optymalizatora na podstawie loss dyskryminatora.
    Jeśli d_loss < target_loss → zmniejsza LR (dyskryminator za mocny)
    Jeśli d_loss > target_loss → zwiększa LR (dyskryminator za słaby)
    """

    def __init__(self, optimizer, target_loss=0.5, factor=0.2, min_lr=0.000001, max_lr=0.001):
        self.optimizer = optimizer
        self.target_loss = target_loss
        self.factor = factor
        self.min_lr = min_lr
        self.max_lr = max_lr

    def step(self, d_loss):
        if d_loss < self.target_loss:
            # zmniejszamy LR
            new_lr = 0.0000001 #max(self.min_lr, self.optimizer.param_groups[0]['lr'] * self.factor)
            print(new_lr)

        elif d_loss < self.target_loss + 0.1 and d_loss > self.target_loss:
            # zwiększamy LR

            new_lr = 0.0001 #min(self.max_lr, self.optimizer.param_groups[0]['lr'] / self.factor)
            print(new_lr)
        else:
            new_lr =  0.001


        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


g_scheduler = StepLR(g_opt, step_size=10, gamma=0.8)  # co 20 epok LR *0.5
#d_scheduler = AdaptiveLRScheduler(d_opt, target_loss=0.3)  # metrics
all_d_losses = []
all_g_losses = []

for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch + 1, num_epochs))
    d_loss, g_loss = train(train_loader, g, d, criterion_l1, criterion_bce, d_opt, g_opt, device)

    all_d_losses.append(d_loss)
    all_g_losses.append(g_loss)

    g_scheduler.step()
    d_scheduler.step(d_loss)

    for gray_images, color_images in val_loader:
        gray_images = gray_images.to(device)
        color_images = color_images.to(device)
        break  # tylko pierwszy batch

    with torch.no_grad():
        fake_imm = g(gray_images)
        imshow(color_images[0], gray_images[0], fake_imm[0])

plot_losses(all_d_losses, all_g_losses)
# zapis wag generatora
torch.save(g.state_dict(), "saved/generator.pth")

# zapis wag dyskryminatora
torch.save(d.state_dict(), "saved/discriminator.pth")