import torch
from torch import Tensor


def gaussian_kernel(x1: Tensor, x2: Tensor, scales: Tensor) -> Tensor:
    """
    Gaussian Kernel, also called Radial-Basis Function.
    Parameters
    ----------
    x1: Tensor of shape (N, D)
    x2: Tensor of shape (M, D)
    scales: Tensor of shape (K,)

    Returns
    -------
    Gaussian distance between x1 and x2
    """
    # x1: (N, D)
    # x2: (M, D)
    # scales: (K,)

    # (N, M, D)
    residuals = x1[:, None] - x2[None, :]

    # (N, M)
    norms = torch.sum(torch.square(residuals), dim=-1)

    # (N, M, K)
    exponent = norms[:, :, None] / (2.0 * scales[None, None, :])

    # (N, M)
    return torch.sum(torch.exp(-exponent), dim=-1)


def mmd_loss(samples: Tensor, target: Tensor, kernel=gaussian_kernel, scales: Tensor | str | int = "auto", reduction: str = "mean"):
    """
    Compute the Maximum-Mean-Discrepancy (MMD) between samples from two distributions
    Parameters
    ----------
    samples: Samples from the target distrbution. Tensor of shape (N, D)
    target: Samples from the actual distribution. Tensor of shape (M, D)
    kernel: Kernel distance function to use
    scales: Characteristic scales used in the kernel function. Tensor of shape (K,)

    Returns
    -------
    mmd: Maximum-Mean-Discrepancy between data and code
    """
    if scales == "auto":
        scales = torch.logspace(-3, 3, 30)
    else:
        scales = scales

    target = target.to(samples.device)
    scales = scales.to(samples.device)

    l1 = torch.mean(kernel(samples, samples, scales=scales), dim=1)
    l2 = torch.mean(kernel(target, target, scales=scales), dim=1)
    l3 = torch.mean(kernel(samples, target, scales=scales), dim=1)

    mmd = l1 + l2 - 2.0 * l3

    match reduction:
        case "none":
            pass
        case "mean":
            mmd = mmd.mean(0)
        case "sum":
            mmd = mmd.sum(0)
        case _:
            raise ValueError(f"Invalid reduction {reduction}")

    return mmd
