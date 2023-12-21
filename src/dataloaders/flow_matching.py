
import torch
from torch.utils.data import DataLoader

from lightning_trainable.metrics import sinkhorn


import src.utils as U


class FlowMatchingDataLoader(DataLoader):
    def __iter__(self):
        return OTCoupledIterator(self)


class OTCoupledIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __iter__(self):
        return self

    def __next__(self):
        x1 = next(self.dataloader)

        batch_size = x1.size(0)

        x0 = torch.randn_like(x1)

        pi = sinkhorn(x0, x1, epsilon=1e-6, steps=1000)
        perm = torch.multinomial(pi, num_samples=1).squeeze(1)
        x1 = x1[perm]

        t = x1.new_rand(batch_size)
        t_expanded = U.unsqueeze_right(t, x1.dim() - 1).expand_as(x1)

        xt = t_expanded * x1 + (1 - t_expanded) * x0

        vstar = x1 - x0

        return xt, x0, x1, t, vstar