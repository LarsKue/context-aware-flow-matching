
import torch
from torch import Tensor


def coverage(x: Tensor, y: Tensor, threshold: float = 0.5) -> Tensor:
    """
    Coverage measures the fraction of point clouds in y that are matched to at least one point cloud in x.
    For each point cloud in x, its nearest neighbor in y is marked as such.
    Parameters
    ----------
    x: Tensor of shape (..., D)
    y: Tensor of shape (..., D)
    threshold: Threshold to use for coverage computation.

    Returns
    -------
    coverage: Tensor of shape (...)
    """
    residuals = x[:, None] - y[None, :]

    pairwise_distances = torch.sum(torch.square(residuals), dim=-1)

    return torch.mean(torch.any(pairwise_distances < threshold, dim=-1))
