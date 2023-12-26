
import torch
from torch.utils.data import Dataset
from torch import Tensor

from lightning_trainable.metrics import sinkhorn


@torch.no_grad()
def optimal_transport(x0: Tensor, x1: Tensor, epsilon: float = 0.01, max_steps: int = 100, atol=1e-3) -> (Tensor, Tensor):
    pi = sinkhorn(x0, x1, epsilon=epsilon, max_steps=max_steps, atol=atol).exp()
    perm = torch.multinomial(pi, num_samples=1).squeeze(1)

    x1 = x1[perm]

    return x0, x1


class ContextAwareFlowMatchingDataset(Dataset):
    def __init__(self, inner: Dataset, subset_size: int):
        super().__init__()

        self.inner = inner
        self.subset_size = subset_size

    def __getitem__(self, item):
        sn1 = self.inner[item]

        sn0 = torch.randn_like(sn1)

        set_size, _ = sn1.shape

        subset = torch.randperm(set_size)[:self.subset_size]
        ssn0, ssn1 = sn0[subset], sn1[subset]

        ssn0, ssn1 = optimal_transport(ssn0, ssn1, epsilon=0.25, max_steps=250, atol=1e-3)

        t = torch.rand(self.subset_size, 1)

        ssnt = t * ssn1 + (1 - t) * ssn0

        vstar = ssn1 - ssn0

        return sn0, sn1, ssn0, ssn1, ssnt, t, vstar

    def __len__(self):
        return len(self.inner)
