from typing import List
from torch import Tensor
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def imshow(real_img: Tensor, gray_img: Tensor, fake_img: Tensor) -> None:
    """Displays real, grayscale, and generated color images side by side."""
    gray_np = gray_img.squeeze().cpu().numpy()
    real_np = (real_img.permute(1, 2, 0).cpu().numpy() + 1) / 2
    fake_np = (fake_img.permute(1, 2, 0).cpu().numpy() + 1) / 2

    plt.figure(figsize=(12, 4))
    for i, (img, title) in enumerate(
        zip([real_np, gray_np, fake_np], ["Real Color", "Grayscale", "Generated Color"])
    ):
        plt.subplot(1, 3, i + 1)
        plt.title(title)
        plt.imshow(img, cmap=None if i != 1 else "gray")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_losses(all_d_losses: List[float], all_g_losses: List[float]) -> None:
    """Plot discriminator and generator training losses."""
    plt.figure(figsize=(10, 5))
    plt.plot(all_d_losses, label="Discriminator Loss")
    plt.plot(all_g_losses, label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def save_colorized_image(img_array: np.ndarray, output_path: str) -> None:
    """Save colorized NumPy image ([0,1]) as PNG/JPG file."""
    img_uint8 = (img_array * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(output_path)