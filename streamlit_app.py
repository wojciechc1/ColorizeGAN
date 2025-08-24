import streamlit as st
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from model import UNet
import os

from PIL import Image

MAX_SIZE = 512

def preprocess_image(img_pil: Image.Image) -> Image.Image:
    w, h = img_pil.size
    if max(w, h) > MAX_SIZE:
        # wyznacz skale
        scale = MAX_SIZE / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img_pil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "saved/generator1.pth"

g = None
if os.path.exists(MODEL_PATH):
    g = UNet().to(device)
    try:
        g.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        g.eval()
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
else:
    st.warning(f"Model file not found at {MODEL_PATH}")

def colorize_image(img_pil: Image.Image) -> np.ndarray:
    try:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
        img_tensor = transform(img_pil.convert("RGB")).unsqueeze(0).to(device)
        gray_tensor = TF.rgb_to_grayscale(img_tensor, num_output_channels=1)

        with torch.no_grad():
            fake_color = g(gray_tensor)
        fake_color = (fake_color.squeeze(0).cpu() + 1) / 2
        return fake_color.permute(1,2,0).numpy()
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# --- Streamlit UI ---
st.title("Grayscale -> Colorizer")
st.info("ℹ️ Due to Streamlit cloud resource limits, it is recommended to use images smaller than 800px. "
        "Larger files are automatically resized, which may affect performance.")


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        img = preprocess_image(img)
        st.image(img, caption="Original (resized)", use_container_width=True)
    except Exception as e:
        st.error(f"Cannot open image: {e}")
        img = None

    if img and st.button("Colorize"):
        if g is not None:
            result = colorize_image(img)
            if result is not None:
                st.image(result, caption="Colorized Image", use_container_width=True)
        else:
            st.error("Model not loaded, cannot colorize.")
