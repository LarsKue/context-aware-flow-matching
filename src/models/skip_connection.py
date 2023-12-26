
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
