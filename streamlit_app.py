import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from model import UNet
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
g = UNet().to(device)
g.load_state_dict(torch.load("saved/generator1.pth", map_location=device))
g.eval()


def colorize_image(img_pil: Image.Image) -> np.ndarray:
    """
    Converts a grayscale or RGB PIL image to a colorized image using the pre-trained UNet.

    Args:
        img_pil (PIL.Image.Image): Input image.

    Returns:
        np.ndarray: Colorized image as a numpy array [H, W, 3] in [0,1].
    """
    # --- Preprocessing ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img_tensor = transform(img_pil.convert("RGB")).unsqueeze(0).to(device)  # [1,3,H,W]

    # Grayscale
    gray_tensor = TF.rgb_to_grayscale(img_tensor, num_output_channels=1)  # [1,1,H,W]

    # --- Generate colorized output ---
    with torch.no_grad():
        fake_color = g(gray_tensor)  # [1,3,H,W]

    # --- Postprocessing ---
    fake_color = (fake_color.squeeze(0).cpu() + 1) / 2  # [3,H,W] -> [0,1]
    fake_np = fake_color.permute(1, 2, 0).numpy()  # [H,W,3]

    return fake_np


import streamlit as st
from PIL import Image

st.title("Grayscale -> Colorizer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Original Image", use_container_width=True)

    if st.button("Colorize"):
        colorized = colorize_image(img)
        st.image(colorized, caption="Colorized Image", use_container_width=True)
