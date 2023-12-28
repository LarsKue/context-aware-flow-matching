
import torch
from torch import Tensor


def jensen_shannon_divergence(x: Tensor, y: Tensor) -> Tensor:
    """
    The Jensen-Shannon divergence is a symmetrized and smoothed version of the Kullback-Leibler divergence.
    Parameters
    ----------
    x: Tensor of shape (..., D)
    y: Tensor of shape (..., D)

    Returns
    -------
    jensen_shannon_divergence: Tensor of shape (...)
    """
    residuals = x[:, None] - y[None, :]

    pairwise_distances = torch.sum(torch.square(residuals), dim=-1)

    first = torch.min(pairwise_distances, dim=-1)[0]
    second = torch.min(pairwise_distances, dim=-2)[0]

    return first + second
