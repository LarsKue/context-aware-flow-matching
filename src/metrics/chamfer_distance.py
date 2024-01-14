
import torch
from torch import Tensor


def chamfer_distance(x: Tensor, y: Tensor) -> Tensor:
    """
    The Chamfer distance is the sum of the minimum squared Euclidean distance
    between each point in x and its nearest neighbor in y.
    Parameters
    ----------
    x: Tensor of shape (..., S, D)
    y: Tensor of shape (..., S, D)

    Returns
    -------
    chamfer_distance: Tensor of shape (...)
    """
    residuals = x[:, None] - y[None, :]

    pairwise_distances = torch.sum(torch.square(residuals), dim=-1)

    first = torch.min(pairwise_distances, dim=0).values
    second = torch.min(pairwise_distances, dim=1).values

    return first.sum(-1) + second.sum(-1)
