
import torch
from torch import Tensor

from torch.utils.data import default_collate

from lightning_trainable.metrics import sinkhorn

import src.utils as U


@torch.no_grad()
def optimal_transport(x0: Tensor, x1: Tensor, epsilon: float = 1e-6, steps: int = 1000) -> (Tensor, Tensor):
    pi = sinkhorn(x0, x1, epsilon=epsilon, steps=steps)
    perm = torch.multinomial(pi, num_samples=1).squeeze(1)

    x1 = x1[perm]

    return x0, x1


@torch.no_grad()
def set_optimal_transport(sn0: Tensor, sn1: Tensor, epsilon: float = 1e-3, steps: int = 100) -> (Tensor, Tensor):
    batch_size, set_size, *_ = sn1.shape

    for b in range(batch_size):
        sn0[b], sn1[b] = optimal_transport(sn0[b], sn1[b], epsilon=epsilon, steps=steps)

    return sn0, sn1


@torch.no_grad()
def cafm_collate(batch: list[Tensor], subset_size: int) -> tuple[Tensor, ...]:
    """
    Optimal-Transport matching for set-based data is non-trivial:
    1. Using OT on the batch dimension as in classical OTFM is wrong,
        since each item in the batch is conditionally independent.
    2. The set size is often prohibitively large for OT matching.
    3. We must use the entire batch to give the set encoder good training signal.
    4. Taking a small subset on the set dimension is legal, but leads to poor matching results.
    5. We cannot use an encoder-decoder structure if we want to sample arbitrary set sizes.

    We take a sufficiently large subset, controlled by a hyperparameter.

    We can do this in the collate function so that it happens in a worker process, thus not slowing training.
    Alternatively, this can be done by using the CAFM dataset wrapper.
    """
    sn1 = default_collate(batch)
    sn1 = U.normalize(sn1, dim=1)

    sn0 = torch.randn_like(sn1)

    batch_size, set_size, *_ = sn1.shape
    device = sn1.device

    subset = torch.randperm(set_size, device=device)[:subset_size]
    ssn0, ssn1 = sn0[:, subset], sn1[:, subset]
    ssn0, ssn1 = set_optimal_transport(ssn0, ssn1)

    t = torch.rand(batch_size, 1, 1, device=device)

    ssnt = t * ssn1 + (1 - t) * ssn0

    vstar = ssn1 - ssn0

    return sn0, sn1, ssn0, ssn1, ssnt, t, vstar
