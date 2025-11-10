import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from src.models.generator import Generator
from typing import Union
from torch import Tensor
import numpy as np
import os


class Colorizer:
    """Handles loading the model and colorizing grayscale images."""

    def __init__(self, model_path: str, device: Union[str, torch.device]) -> None:
        """Initialize device, load model, and set up preprocessing."""
        self.device: torch.device = self._setup_device(device)
        self.model: torch.nn.Module = self._load_model(model_path)
        self.transform: transforms.Compose = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _setup_device(self, device: Union[str, torch.device]) -> torch.device:
        """Set device, fallback to CPU if CUDA unavailable."""
        device = torch.device(device)
        if "cuda" in str(device) and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU.")
            device = torch.device("cpu")
        return device

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load pretrained Generator model from disk."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = Generator().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def _preprocess(self, image: Union[str, Image.Image]) -> Tensor:
        """Load and prepare image for inference. Accepts path or PIL Image."""
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found: {image}")
            img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise TypeError("image must be a file path or PIL.Image.Image")

        img_tensor: Tensor = (
            self.transform(img).unsqueeze(0).to(self.device)
        )  # [1,3,H,W]
        gray_tensor: Tensor = TF.rgb_to_grayscale(img_tensor, num_output_channels=1)
        return gray_tensor

    def _inference(self, gray_tensor: torch.Tensor) -> Tensor:
        """Run model inference without gradients."""
        with torch.no_grad():
            return self.model(gray_tensor)

    def _postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert model output to a NumPy image."""
        tensor = (tensor.squeeze(0).cpu() + 1) / 2  # [3,H,W] > [0,1]
        img_np: np.ndarray = tensor.permute(1, 2, 0).numpy()
        return img_np

    def __call__(self, image: Union[str, Image.Image]) -> np.ndarray:
        """Full colorization pipeline: preprocess > infer > postprocess."""
        gray: Tensor = self._preprocess(image)
        output: Tensor = self._inference(gray)
        return self._postprocess(output)
