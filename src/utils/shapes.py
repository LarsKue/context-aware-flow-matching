
import torch

from torch import Tensor


def expand_dim(x: Tensor, dim: int, k: int) -> Tensor:
    """
    Expand x along dim by k times

    Example:
        >>> x = torch.randn(3, 4)
        >>> x.shape
        torch.Size([3, 4])
        >>> expand_dim(x, 0, 2).shape
        torch.Size([6, 4])
    """
    expand = [-1] * x.dim()
    expand[dim] = k
    return x.expand(*expand)


def unsqueeze_right(x: Tensor, k: int) -> Tensor:
    """
    Unsqueeze x k times to the right

    Example:
        >>> x = torch.randn(3, 4)
        >>> x.shape
        torch.Size([3, 4])
        >>> unsqueeze_right(x, 2).shape
        torch.Size([3, 4, 1, 1])
    """
    unsqueeze = [slice(None)] * x.dim()
    unsqueeze += [None] * k
    return x[unsqueeze]
