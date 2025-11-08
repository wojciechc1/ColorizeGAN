import torch
import numpy as np
from PIL import Image
from src.models.generator import Generator
from src.inference.colorizer import Colorizer


def test_colorizer_call(tmp_path):
    model_file = tmp_path / "test_generator.pth"
    img_file = tmp_path / "test_img.png"

    dummy_gen = Generator()
    torch.save(dummy_gen.state_dict(), model_file)

    colorizer = Colorizer(model_path=str(model_file), device="cpu")

    img = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8))
    img.save(img_file)

    out = colorizer(str(img_file))
    assert isinstance(out, np.ndarray)
    assert out.shape[2] == 3
