
import torch
import torch.nn as nn

from torch import Tensor


from .mab import MultiheadAttentionBlock


class InducedSetAttentionBlock(nn.Module):
    """
    Induced Set-Attention Block as described in arXiv:1810.00825
    This block is similar to the SetAttentionBlock, but alleviates the quadratic time complexity by using
    a separate set of inducing points ("seeds") which are attended to with the input set.
    """
    def __init__(self, in_features: int, out_features: int, heads: int = 4, seeds: int = 32):
        super().__init__()
        self.register_buffer("in_features", torch.tensor(in_features, dtype=torch.int64))
        self.register_buffer("out_features", torch.tensor(out_features, dtype=torch.int64))
        self.register_buffer("heads", torch.tensor(heads, dtype=torch.int64))

        self.seeds = nn.Parameter(torch.empty(1, seeds, out_features))
        nn.init.xavier_uniform_(self.seeds)

        self.mab1 = MultiheadAttentionBlock(out_features, in_features, out_features, heads=heads)
        self.mab2 = MultiheadAttentionBlock(in_features, out_features, out_features, heads=heads)

    def forward(self, x: Tensor) -> Tensor:
        # Eq. 9 and 10
        seeds = self.seeds.repeat(x.shape[0], 1, 1)
        h = self.mab1(seeds, x)

        return self.mab2(x, h)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features.item()}, out_features={self.out_features.item()}, heads={self.heads.item()}, seeds={self.seeds.size(0)}"
