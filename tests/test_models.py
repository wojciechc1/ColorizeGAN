import torch
from src.models.generator import Generator
from src.models.discriminator import Discriminator


def test_generator_forward():
    g = Generator()
    x = torch.randn(2, 1, 32, 32)
    out = g(x)
    assert out.shape == (2, 3, 32, 32)


def test_discriminator_forward():
    d = Discriminator()
    gray = torch.randn(2, 1, 32, 32)
    color = torch.randn(2, 3, 32, 32)
    out = d(gray, color)
    assert isinstance(out, torch.Tensor)
    assert out.ndim == 1
