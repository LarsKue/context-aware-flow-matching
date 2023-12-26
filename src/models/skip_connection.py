
import torch.nn as nn


class SkipConnection(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()

        self.inner = inner

    def forward(self, x):
        return self.inner(x) + x


class SkipLinear(SkipConnection):
    def __init__(self, features: int):
        super().__init__(nn.Linear(features, features))


class ResidualLayer(nn.Module):
    def __init__(self, features: int, dropout: float = None):
        super().__init__()

        self.linear = nn.Linear(features, features)
        self.relu = nn.ReLU()

        if dropout is not None and dropout > 0.0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = self.linear(x)
        x = self.relu(x)

        return x


class ResidualBlock(nn.Sequential):
    def __init__(self, features: int, layers: int, dropout: float = None):
        layers = [ResidualLayer(features, dropout) for _ in range(layers)]

        super().__init__(*layers)
