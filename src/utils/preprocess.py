
import torch
from torch import Tensor

from .shapes import unsqueeze_right


def normalize(x: Tensor, dim: int) -> Tensor:
    """
    Normalize x to zero mean and unit variance along dim
    """
    mean = torch.mean(x, dim=dim, keepdim=True)
    std = torch.std(x, dim=dim, keepdim=True)

    return (x - mean) / std


def interpolate(x: Tensor, y: Tensor, steps: int) -> Tensor:
    """
    Linearly interpolate between x and y

    @param x: Tensor of shape (n, ...)
    @param y: Tensor of shape (n, ...)
    @param steps: Number of interpolation steps

    @return: Tensor of shape (n, steps, ...)
    """
    t = torch.linspace(0.0, 1.0, steps, device=x.device)
    t = unsqueeze_right(t, x.dim() - 1)
    return t[None, :] * y[:, None] + (1 - t[None, :]) * x[:, None]
