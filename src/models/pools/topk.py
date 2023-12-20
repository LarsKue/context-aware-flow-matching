
import torch
import torch.nn as nn

from torch import Tensor


class TopK(nn.Module):
    def __init__(self, k: int, largest: bool = True, dim: int = 1):
        super().__init__()
        self.register_buffer("k", torch.tensor(k))
        self.register_buffer("largest", torch.tensor(largest))
        self.register_buffer("dim", torch.tensor(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.topk(self.k.item(), dim=self.dim.item(), largest=self.largest.item()).values

    def extra_repr(self) -> str:
        return f"k={self.k.item()}, largest={self.largest.item()}, dim={self.dim.item()}"
