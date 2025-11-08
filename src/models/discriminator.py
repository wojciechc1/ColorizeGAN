import torch
import torch.nn as nn
from torch import Tensor


class Discriminator(nn.Module):
    def __init__(self, dropout_prob: float = 0.05) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(4, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_prob),
            nn.utils.spectral_norm(nn.Conv2d(256, 512, 4, 1, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def forward(self, gray_img: Tensor, color_img: Tensor) -> Tensor:
        x = torch.cat([gray_img, color_img], dim=1)  # [B, 4, H, W]
        return self.model(x).view(
            -1
        )  # flatten do [B], scalar na batch self.model(x).view(-1)  # flatten do [B], scalar na batch
