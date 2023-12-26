
import torch

from torch import Tensor


def repeat_dim(x: Tensor, dim: int, k: int) -> Tensor:
    """
    Repeat x along dim by k times

    Example:
        >>> x = torch.randn(3, 4)
        >>> x.shape
        torch.Size([3, 4])
        >>> repeat_dim(x, 0, 6).shape
        torch.Size([18, 4])
    """
    repeat = [1] * x.dim()
    repeat[dim] = k
    return x.repeat(*repeat)


def expand_dim(x: Tensor, dim: int, k: int) -> Tensor:
    """
    Expand x along dim by k times

    Example:
        >>> x = torch.randn(3, 4)
        >>> x.shape
        torch.Size([3, 4])
        >>> expand_dim(x, 0, 6).shape
        torch.Size([6, 4])
    """
    expand = [-1] * x.dim()
    expand[dim] = k
    return x.expand(*expand)


def unsqueeze_dim(x: Tensor, dim: int, k: int) -> Tensor:
    """
    Unsqueeze x along dim by k times

    Example:
        >>> x = torch.randn(3, 4)
        >>> x.shape
        torch.Size([3, 4])
        >>> unsqueeze(x, 1, 2).shape
        torch.Size([3, 1, 1, 4])
    """
    unsqueeze = [slice(None)] * x.dim()
    unsqueeze[dim:dim] = [None] * k
    return x[unsqueeze]


def unsqueeze_left(x: Tensor, k: int) -> Tensor:
    """
    Unsqueeze x k times to the left

    Example:
        >>> x = torch.randn(3, 4)
        >>> x.shape
        torch.Size([3, 4])
        >>> unsqueeze_left(x, 2).shape
        torch.Size([1, 1, 3, 4])
    """
    return unsqueeze_dim(x, 0, k)


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
    return unsqueeze_dim(x, x.dim(), k)


def mean_except(x: Tensor, dim: int) -> Tensor:
    """
    Take the mean of x over all dimensions except dim

    Example:
        >>> x = torch.randn(3, 4, 5)
        >>> x.shape
        torch.Size([3, 4, 5])
        >>> mean_except(x, 0).shape
        torch.Size([3])
    """
    dims = list(range(x.dim()))
    dims.remove(dim)
    return x.mean(dims)
