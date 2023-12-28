
import torch
import torch.nn as nn
from torch import Tensor


class SkipConnection(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()

        self.inner = inner

    def forward(self, x):
        return self.inner(x) + x


class SkipLinear(SkipConnection):
    def __init__(self, features: int):
        super().__init__(nn.Linear(features, features))


class ResidualLayer(SkipConnection):
    def __init__(self, features: int):
        inner = nn.Sequential(nn.Linear(features, features), nn.SELU())

        super().__init__(inner)


class ResidualBlock(nn.Sequential):
    def __init__(self, features: int, layers: int):
        layers = [ResidualLayer(features) for _ in range(layers)]

        super().__init__(*layers)
