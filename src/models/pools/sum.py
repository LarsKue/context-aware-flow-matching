
import torch
import torch.nn as nn

from torch import Tensor


class Sum(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.register_buffer("dim", torch.tensor(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.sum(dim=self.dim.item(), keepdim=True)

    def extra_repr(self) -> str:
        return f"dim={self.dim.item()}"
