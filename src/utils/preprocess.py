
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


def rotate(x: Tensor, angles: Tensor) -> Tensor:
    """
    Rotate x by the given euler angles

    @param x. Tensor of shape (..., 3)
    @param angles: Tensor of shape (..., 3)

    @return: Tensor of shape (..., 3)
    """
    *batch_shape, dims = x.shape

    if dims != 3:
        raise ValueError(f"Expected tensor of shape (..., 3), got {x.shape}")
    if angles.shape != (*batch_shape, 3):
        raise ValueError(f"Expected tensor of shape {(*batch_shape, 3)}, got {angles.shape}")

    angles = torch.deg2rad(angles)

    rotations = torch.empty((*batch_shape, 3, 3), device=x.device)

    alpha, beta, gamma = torch.unbind(angles, dim=-1)
    rotations[..., 0, 0] = torch.cos(beta) * torch.cos(gamma)
    rotations[..., 0, 1] = torch.cos(beta) * torch.sin(gamma)
    rotations[..., 0, 2] = -torch.sin(beta)
    rotations[..., 1, 0] = torch.sin(alpha) * torch.sin(beta) * torch.cos(gamma) - torch.cos(alpha) * torch.sin(gamma)
    rotations[..., 1, 1] = torch.sin(alpha) * torch.sin(beta) * torch.sin(gamma) + torch.cos(alpha) * torch.cos(gamma)
    rotations[..., 1, 2] = torch.sin(alpha) * torch.cos(beta)
    rotations[..., 2, 0] = torch.cos(alpha) * torch.sin(beta) * torch.cos(gamma) + torch.sin(alpha) * torch.sin(gamma)
    rotations[..., 2, 1] = torch.cos(alpha) * torch.sin(beta) * torch.sin(gamma) - torch.sin(alpha) * torch.cos(gamma)
    rotations[..., 2, 2] = torch.cos(alpha) * torch.cos(beta)

    return torch.einsum("...ij,...j->...i", rotations, x)
