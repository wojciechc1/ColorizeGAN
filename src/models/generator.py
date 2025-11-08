import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Encoder
        self.enc1 = self.block(1, 64)
        self.enc2 = self.block(64, 128)
        self.enc3 = self.block(128, 256)
        # Bottleneck
        self.bottleneck = nn.Sequential(
            self.block(256, 512),
            # nn.Dropout2d(0.1)
        )
        # Decoder
        self.dec3 = self.block(512 + 256, 256)
        self.dec2 = self.block(256 + 128, 128)
        self.dec1 = self.block(128 + 64, 64)
        # Output
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)

    def block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.interpolate(e1, scale_factor=0.5))
        e3 = self.enc3(F.interpolate(e2, scale_factor=0.5))

        # Bottleneck
        b = self.bottleneck(F.interpolate(e3, scale_factor=0.5))

        # Decoder
        d3 = F.interpolate(b, size=e3.shape[2:])
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, size=e2.shape[2:])
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, size=e1.shape[2:])
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = torch.tanh(self.out_conv(d1))
        return out
