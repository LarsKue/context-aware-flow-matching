
import torch
import torch.nn as nn

from torch import Tensor

from ..transforms import MultiheadAttentionBlock


class PoolingByMultiheadAttention(nn.Module):
    """
    Pooling by Multi-Head Attention as described in arXiv:1810.00825
    This block applies Multi-Head Attention to the input set with a learnable set of seed vectors.
    For an input tensor of shape (batch_size, set_size, input_features),
    the output tensor will have shape (batch_size, seeds, output_features).
    """
    def __init__(self, input_features: int, hidden_features: int, output_features: int, heads: int, seeds: int):
        super().__init__()
        self.register_buffer("input_features", torch.tensor(input_features, dtype=torch.int64))
        self.register_buffer("hidden_features", torch.tensor(hidden_features, dtype=torch.int64))
        self.register_buffer("output_features", torch.tensor(output_features, dtype=torch.int64))
        self.register_buffer("heads", torch.tensor(heads, dtype=torch.int64))
        self.register_buffer("seeds", torch.tensor(seeds, dtype=torch.int64))

        self.S = nn.Parameter(torch.empty(1, seeds, input_features))
        nn.init.xavier_uniform_(self.S)

        self.mab = MultiheadAttentionBlock(input_features, hidden_features, output_features, heads=heads)

        self.rFF = nn.Linear(input_features, hidden_features)
        nn.init.xavier_uniform_(self.rFF.weight)
        self.rFF.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        # Eq. 11
        seeds = self.S.repeat(x.shape[0], 1, 1)
        return self.mab(seeds, self.rFF(x))

    def extra_repr(self) -> str:
        return (f"input_features={self.input_features.item()}, "
                f"hidden_features={self.hidden_features.item()}, "
                f"output_features={self.output_features.item()}, "
                f"heads={self.heads.item()}, "
                f"seeds={self.seeds.item()}")
