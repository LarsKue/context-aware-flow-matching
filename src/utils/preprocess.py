
import torch
from torch import Tensor


def normalize(x: Tensor, dim: int) -> Tensor:
    """
    Normalize x to zero mean and unit variance along dim
    """
    mean = torch.mean(x, dim=dim, keepdim=True)
    std = torch.std(x, dim=dim, keepdim=True)

    return (x - mean) / std
