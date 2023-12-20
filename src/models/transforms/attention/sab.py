
import torch
import torch.nn as nn

from torch import Tensor


from .mab import MultiheadAttentionBlock


class SetAttentionBlock(nn.Module):
    """
    Set-Attention Block as described in arXiv:1810.00825
    Performs Self-Attention between elements of a set, resulting in a set of equal size.
    This encodes information about pairwise interactions among the elements in the input set.
    """

    def __init__(self, in_features: int, out_features: int, heads: int = 4):
        super().__init__()
        self.register_buffer("in_features", torch.tensor(in_features, dtype=torch.int64))
        self.register_buffer("out_features", torch.tensor(out_features, dtype=torch.int64))
        self.register_buffer("heads", torch.tensor(heads, dtype=torch.int64))

        self.mab = MultiheadAttentionBlock(in_features, in_features, out_features, heads=heads)

    def forward(self, x: Tensor) -> Tensor:
        # Eq. 8
        return self.mab(x, x)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features.item()}, out_features={self.out_features.item()}, heads={self.heads.item()}"
