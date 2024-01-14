
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential


class CheckpointedSequential(nn.Sequential):
    def __init__(self, *layers: nn.Module, segments: int):
        super().__init__(*layers)

        self.register_buffer("segments", torch.tensor(segments, dtype=torch.int64))

    def forward(self, x):
        if not self.training:
            return super().forward(x)
        
        return checkpoint_sequential(self, self.segments.item(), x, use_reentrant=False)

    @classmethod
    def _layers_from_nested(cls, sequential: nn.Sequential):
        layers = []
        for layer in sequential:
            if isinstance(layer, nn.Sequential):
                layers.extend(cls._layers_from_nested(layer))
            else:
                layers.append(layer)

        return layers

    @classmethod
    def from_nested(cls, sequential: nn.Sequential, segments: int):
        layers = cls._layers_from_nested(sequential)
        return cls(*layers, segments=segments)
