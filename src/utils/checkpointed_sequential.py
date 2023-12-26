
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential


class CheckpointedSequential(nn.Sequential):
    def __init__(self, *layers: nn.Module, segments: int):
        super().__init__(*layers)

        self.register_buffer("segments", torch.tensor(segments, dtype=torch.int64))

    def forward(self, x):
        return checkpoint_sequential(self, self.segments.item(), x, use_reentrant=False)
