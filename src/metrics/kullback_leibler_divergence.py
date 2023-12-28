
import torch
from torch import Tensor


def kullback_leibler_divergence(x: Tensor, y: Tensor) -> Tensor:
    """
    The Kullback-Leibler divergence is a measure of how one probability distribution diverges from a second expected
    probability distribution.
    Parameters
    ----------
    x: Tensor of shape (..., D)
    y: Tensor of shape (..., D)

    Returns
    -------
    kullback_leibler_divergence: Tensor of shape (...)
    """
    raise NotImplementedError()
