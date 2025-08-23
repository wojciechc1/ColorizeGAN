import torch
import torch.nn.utils as nn_utils

def train(train_loader, g, d, criterion_l1, criterion_bce, d_opt, g_opt, device):
    total_loss_d = 0
    total_loss_g = 0


    for i, (gray_images, color_images) in enumerate(train_loader):
        # przenieś batch na device
        gray_images = gray_images.to(device)
        color_images = color_images.to(device)

        # -----------------------------
        # Trening dyskryminatora
        # -----------------------------
        fake_color_images = g(gray_images)  # shape [B, 1, 32, 32]

        real_preds = d(gray_images, color_images)
        fake_preds = d(gray_images, fake_color_images.detach())

        # d loss
        real_labels = torch.full_like(real_preds, 0.9, device=device)
        fake_labels = torch.zeros_like(fake_preds, device=device)

        d_loss_real = criterion_bce(real_preds, real_labels)
        d_loss_fake = criterion_bce(fake_preds, fake_labels)
        d_loss = (d_loss_real + d_loss_fake) / 2

        # backward dla D
        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()

        # -----------------------------
        # Trening generatora
        # -----------------------------
        fake_color_images = g(gray_images)
        fake_preds = d(gray_images, fake_color_images)

        real_labels = torch.ones_like(fake_preds, device=device)

        loss_gan = criterion_bce(fake_preds, real_labels)  # GAN loss
        loss_pixel = criterion_l1(fake_color_images, color_images)  # L1 pixel loss

        g_loss = loss_gan + 30 * loss_pixel

        # backward dla G
        g_opt.zero_grad()
        g_loss.backward()
        #nn_utils.clip_grad_norm_(g.parameters(), max_norm=1.0)
        g_opt.step()

        # zbieranie statystyk
        total_loss_d += d_loss.item()
        total_loss_g += g_loss.item()
        if i % 10 == 0:
          print("g_loss", g_loss.item(), "d_loss", d_loss.item())

    avg_d_loss = total_loss_d / len(train_loader)
    avg_g_loss = total_loss_g / len(train_loader)

    return avg_d_loss, avg_g_loss
