import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder (downsampling)
        self.enc1 = self.block(1, 64)     # grayscale → 64
        self.enc2 = self.block(64, 128)
        self.enc3 = self.block(128, 256)

        # Bottleneck
        self.bottleneck = self.block(256, 512)

        # Decoder (upsampling + skip connections)
        self.up1 = self.up_block(512, 256)
        self.dec1 = self.block(512, 256)  # 256 from up, 256 from skip

        self.up2 = self.up_block(256, 128)
        self.dec2 = self.block(256, 128)

        self.up3 = self.up_block(128, 64)
        self.dec3 = self.block(128, 64)

        # Output layer
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)  # RGB output

    def block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def up_block(self, in_ch, out_ch):
        return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)           # 64x32x32
        e2 = self.enc2(nn.MaxPool2d(2)(e1))  # 128x16x16
        e3 = self.enc3(nn.MaxPool2d(2)(e2))  # 256x8x8

        # Bottleneck
        b = self.bottleneck(nn.MaxPool2d(2)(e3))  # 512x4x4

        # Decoder
        d1 = self.up1(b)           # 256x8x8
        d1 = self.dec1(torch.cat([d1, e3], dim=1))  # skip connection

        d2 = self.up2(d1)          # 128x16x16
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d3 = self.up3(d2)          # 64x32x32
        d3 = self.dec3(torch.cat([d3, e1], dim=1))

        out = torch.tanh(self.out_conv(d3))  # output RGB in [-1, 1]
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 1)  # output: logity
        )

    def forward(self, gray_img, color_img):
        x = torch.cat([gray_img, color_img], dim=1)  # [B, 4, H, W]
        return self.model(x).view(-1)  # flatten do [B], scalar na batch