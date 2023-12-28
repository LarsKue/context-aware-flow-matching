
import torch
from torch import Tensor


def chamfer_distance(x: Tensor, y: Tensor) -> Tensor:
    """
    The Chamfer distance is the sum of the minimum squared Euclidean distance
    between each point in x and its nearest neighbor in y.
    Parameters
    ----------
    x: Tensor of shape (..., D)
    y: Tensor of shape (..., D)

    Returns
    -------
    chamfer_distance: Tensor of shape (...)
    """
    residuals = x[:, None] - y[None, :]

    pairwise_distances = torch.sum(torch.square(residuals), dim=-1)

    first = torch.min(pairwise_distances, dim=-1)[0]
    second = torch.min(pairwise_distances, dim=-2)[0]

    return first + second
