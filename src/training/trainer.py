import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple


class Trainer:
    """Trainer for GAN-based colorization model."""

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        g_optimizer: torch.optim.Optimizer,
        d_optimizer: torch.optim.Optimizer,
        criterion_l1: nn.Module,
        criterion_bce: nn.Module,
        device: torch.device,
        l1_weight: float = 30.0,
    ) -> None:
        self.g = generator.to(device)
        self.d = discriminator.to(device)
        self.g_opt = g_optimizer
        self.d_opt = d_optimizer
        self.criterion_l1 = criterion_l1
        self.criterion_bce = criterion_bce
        self.device = device
        self.l1_weight = l1_weight

    def train_epoch(
        self, train_loader: DataLoader, log_interval: int = 10
    ) -> Tuple[float, float]:
        """Train models for one epoch."""
        self.g.train()
        self.d.train()

        total_loss_d = 0.0
        total_loss_g = 0.0

        for i, (gray_images, color_images) in enumerate(train_loader):
            gray_images = gray_images.to(self.device)
            color_images = color_images.to(self.device)

            # ---------- Train Discriminator ----------
            fake_color = self.g(gray_images)
            real_preds = self.d(gray_images, color_images)
            fake_preds = self.d(gray_images, fake_color.detach())

            real_labels = torch.full_like(real_preds, 0.9, device=self.device)
            fake_labels = torch.zeros_like(fake_preds, device=self.device)

            d_loss_real = self.criterion_bce(real_preds, real_labels)
            d_loss_fake = self.criterion_bce(fake_preds, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2

            self.d_opt.zero_grad()
            d_loss.backward()
            self.d_opt.step()

            # ---------- Train Generator ----------
            fake_color = self.g(gray_images)
            fake_preds = self.d(gray_images, fake_color)

            real_labels = torch.ones_like(fake_preds, device=self.device)

            loss_gan = self.criterion_bce(fake_preds, real_labels)
            loss_pixel = self.criterion_l1(fake_color, color_images)
            g_loss = loss_gan + self.l1_weight * loss_pixel

            self.g_opt.zero_grad()
            g_loss.backward()
            # nn_utils.clip_grad_norm_(self.g.parameters(), max_norm=1.0)
            self.g_opt.step()

            total_loss_d += d_loss.item()
            total_loss_g += g_loss.item()

            if i % log_interval == 0:
                print(
                    f"[Batch {i}] g_loss: {g_loss.item():.4f}, d_loss: {d_loss.item():.4f}"
                )

        avg_d_loss = total_loss_d / len(train_loader)
        avg_g_loss = total_loss_g / len(train_loader)

        return avg_d_loss, avg_g_loss
