import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from src.config import CONFIG
from src.utils.visualize import imshow, plot_losses
from src.data.dataset import get_dataloader
from src.models.generator import Generator
from src.models.discriminator import Discriminator
from trainer import Trainer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader = get_dataloader(
        CONFIG["batch_size"], CONFIG["train_samples"], CONFIG["val_samples"]
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Models, losses, optimizers
    generator = Generator().to(device)
    discriminator = Discriminator(dropout_prob=CONFIG["dropout"]).to(device)

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    g_opt = torch.optim.Adam(
        generator.parameters(), lr=CONFIG["lr"], betas=CONFIG["betas"]
    )
    d_opt = torch.optim.Adam(
        discriminator.parameters(), lr=CONFIG["lr"], betas=CONFIG["betas"]
    )

    g_scheduler = StepLR(
        g_opt, step_size=CONFIG["scheduler_step"], gamma=CONFIG["scheduler_gamma"]
    )
    d_scheduler = StepLR(
        d_opt, step_size=CONFIG["scheduler_step"], gamma=CONFIG["scheduler_gamma"]
    )

    # Trainer
    trainer = Trainer(
        generator,
        discriminator,
        g_opt,
        d_opt,
        criterion_l1,
        criterion_bce,
        device,
        l1_weight=CONFIG["l1_weight"],
    )

    all_d_losses, all_g_losses = [], []

    # Training loop
    for epoch in range(1, CONFIG["num_epochs"] + 1):
        torch.cuda.empty_cache()
        print(f"Epoch {epoch}/{CONFIG["num_epochs"]}")

        d_loss, g_loss = trainer.train_epoch(train_loader, CONFIG["log_interval"])
        all_d_losses.append(d_loss)
        all_g_losses.append(g_loss)

        g_scheduler.step()
        d_scheduler.step()

        # Visualize first batch from validation set
        for gray_images, color_images in val_loader:
            gray_images = gray_images.to(device)
            color_images = color_images.to(device)
            with torch.no_grad():
                fake_imgs = generator(gray_images)
                imshow(color_images[0], gray_images[0], fake_imgs[0])
            break

    # Plot losses
    plot_losses(all_d_losses, all_g_losses)

    # Save weights
    torch.save(generator.state_dict(), CONFIG["generator_save_path"])
    torch.save(discriminator.state_dict(), CONFIG["discriminator_save_path"])


if __name__ == "__main__":
    main()
