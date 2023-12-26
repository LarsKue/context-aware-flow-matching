
import torch
import torch.nn as nn

from torch import Size, Tensor

from functools import partial

from lightning_trainable import Trainable, TrainableHParams
from lightning_trainable.hparams import Range

import src.losses as L
import src.utils as U

from src.callbacks import SampleCallback
from src.datasets import cafm_collate, ContextAwareFlowMatchingDataset as CAFMDataset

from . import pools
from .skip_connection import SkipLinear


class ModelHParams(TrainableHParams):
    subset_size: int

    features: int = 3
    conditions: int = 1
    embeddings: int = 256

    gamma: Range(0.0, 1.0) = 0.5


class Model(Trainable):
    hparams: ModelHParams

    def __init__(self, hparams, *datasets):
        hparams = self.hparams_type(**hparams)
        datasets = [CAFMDataset(d, hparams.subset_size) for d in datasets]

        super().__init__(hparams, *datasets)

        self.encoder = nn.Sequential(
            nn.Sequential(nn.Linear(self.hparams.features, 64), nn.ReLU()),
            *[nn.Sequential(SkipLinear(64), nn.ReLU()) for _ in range(8)],

            SkipLinear(64),
            pools.TopK(k=512, dim=1),

            nn.Sequential(nn.Linear(64, 128), nn.ReLU()),
            *[nn.Sequential(SkipLinear(128), nn.ReLU()) for _ in range(8)],

            SkipLinear(128),
            pools.TopK(k=128, dim=1),

            nn.Sequential(nn.Linear(128, 256), nn.ReLU()),
            *[nn.Sequential(SkipLinear(256), nn.ReLU()) for _ in range(8)],

            SkipLinear(256),
            pools.TopK(k=32, dim=1),

            nn.Sequential(nn.Linear(256, 512), nn.ReLU()),
            *[nn.Sequential(SkipLinear(512), nn.ReLU()) for _ in range(8)],

            SkipLinear(512),
            pools.TopK(k=8, dim=1),

            nn.Sequential(nn.Linear(512, 1024), nn.ReLU()),
            *[nn.Sequential(SkipLinear(1024), nn.ReLU()) for _ in range(8)],

            SkipLinear(1024),
            pools.Mean(dim=1),
            nn.Flatten(),

            *[nn.Sequential(SkipLinear(1024), nn.ReLU()) for _ in range(16)],
            nn.Linear(1024, self.hparams.embeddings),
        )

        self.flow = nn.Sequential(
            nn.Sequential(nn.Linear(self.hparams.features + self.hparams.conditions + self.hparams.embeddings, 512), nn.ReLU()),
            *[nn.Sequential(SkipLinear(512), nn.ReLU()) for _ in range(8)],

            nn.Sequential(nn.Linear(512, 1024), nn.ReLU()),
            *[nn.Sequential(SkipLinear(1024), nn.ReLU()) for _ in range(8)],

            nn.Sequential(nn.Linear(1024, 512), nn.ReLU()),
            *[nn.Sequential(SkipLinear(512), nn.ReLU()) for _ in range(8)],

            nn.Sequential(nn.Linear(512, 256), nn.ReLU()),
            *[nn.Sequential(SkipLinear(256), nn.ReLU()) for _ in range(8)],

            nn.Sequential(nn.Linear(256, 128), nn.ReLU()),
            *[nn.Sequential(SkipLinear(128), nn.ReLU()) for _ in range(8)],

            nn.Sequential(nn.Linear(128, 64), nn.ReLU()),
            *[nn.Sequential(SkipLinear(64), nn.ReLU()) for _ in range(8)],

            nn.Linear(64, self.hparams.features),
        )

        # initialize as random projection
        nn.init.xavier_normal_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

        # initialize as random drift
        nn.init.xavier_normal_(self.flow[-1].weight)
        nn.init.zeros_(self.flow[-1].bias)

    def compute_metrics(self, batch, batch_idx):
        sn0, sn1, ssn0, ssn1, ssnt, t, vstar = batch

        c = self.encoder(sn1)
        mmd = L.mmd_loss(c, torch.randn_like(c), scales=torch.logspace(-20, 20, base=2, steps=41))

        v = self.velocity(ssnt, t, c)

        mse = U.mean_except(torch.square(v - vstar), 0)

        loss = self.hparams.gamma * mmd + (1 - self.hparams.gamma) * mse

        loss = loss.mean(0)
        mse = mse.mean(0)
        mmd = mmd.mean(0)

        return dict(
            loss=loss,
            mse=mse,
            mmd=mmd,
        )

    def velocity(self, sn: Tensor, t: Tensor | float, c: Tensor) -> Tensor:
        batch_size, set_size, _ = sn.shape

        if isinstance(t, float):
            t = torch.full((batch_size,), fill_value=t, device=self.device)

        t = U.unsqueeze_right(t, sn.dim() - t.dim())
        t = U.expand_dim(t, 1, set_size)

        c = c.unsqueeze(1)
        c = U.expand_dim(c, 1, set_size)

        sntc = torch.cat([sn, t, c], dim=2)

        return self.flow(sntc)

    @torch.no_grad()
    def sample(self, sample_shape: Size = (1,), steps: int = 100) -> Tensor:
        c = torch.randn(sample_shape[0], self.hparams.embeddings, device=self.device)
        sn = torch.randn(sample_shape[0], sample_shape[1], self.hparams.features, device=self.device)

        return self.sample_from(sn, c, steps=steps)

    @torch.no_grad()
    def sample_from(self, noise: Tensor, condition: Tensor, steps: int = 100):
        t = 0.0
        dt = 1.0 / steps

        for i in range(steps):
            velocity = self.velocity(noise, t, condition)
            noise = noise + velocity * dt
            t = t + dt

        return noise

    def train_dataloader(self):
        dl = super().train_dataloader()
        # dl.collate_fn = partial(cafm_collate, subset_size=self.hparams.subset_size)
        return dl

    def val_dataloader(self):
        dl = super().val_dataloader()
        # dl.collate_fn = partial(cafm_collate, subset_size=self.hparams.subset_size)
        return dl

    def test_dataloader(self):
        dl = super().test_dataloader()
        # dl.collate_fn = partial(cafm_collate, subset_size=self.hparams.subset_size)
        return dl

    def configure_callbacks(self):
        callbacks = super().configure_callbacks()
        callbacks.append(SampleCallback())
        return callbacks
