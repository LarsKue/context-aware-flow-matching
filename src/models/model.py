
import torch
import torch.nn as nn

from torch import Size, Tensor

from functools import partial

from lightning_trainable import Trainable, TrainableHParams
from lightning_trainable.hparams import Range

import src.losses as L
import src.utils as U

from src.callbacks import SampleCallback
from src.datasets import cafm_collate

from . import pools


class ModelHParams(TrainableHParams):
    features: int = 3
    conditions: int = 1
    embeddings: int = 256

    gamma: Range(0.0, 1.0) = 0.5


class Model(Trainable):
    hparams: ModelHParams

    def __init__(self, hparams, *datasets):
        super().__init__(hparams, *datasets)

        # self.encoder = nn.Sequential(
        #     nn.Sequential(nn.Linear(self.hparams.features, 64), nn.ReLU()),
        #     *[nn.Sequential(nn.Linear(64, 64), nn.ReLU()) for _ in range(8)],
        #     pools.TopK(k=512, dim=1),
        #     nn.Sequential(nn.Linear(64, 256), nn.ReLU()),
        #     *[nn.Sequential(nn.Linear(256, 256), nn.ReLU()) for _ in range(16)],
        #     pools.TopK(k=64, dim=1),
        #     nn.Sequential(nn.Linear(256, 1024), nn.ReLU()),
        #     *[nn.Sequential(nn.Linear(1024, 1024), nn.ReLU()) for _ in range(16)],
        #     pools.Mean(dim=1),
        #     nn.Flatten(),
        #     *[nn.Sequential(nn.Linear(1024, 1024), nn.ReLU()) for _ in range(8)],
        #     nn.Sequential(nn.Linear(1024, self.hparams.embeddings)),
        # )
        #
        # # diamond shape flow matching model
        # self.flow = nn.Sequential(
        #     nn.Sequential(nn.Linear(self.hparams.features + self.hparams.conditions + self.hparams.embeddings, 64), nn.ReLU()),
        #     *[nn.Sequential(nn.Linear(64, 64), nn.ReLU()) for _ in range(8)],
        #     nn.Sequential(nn.Linear(64, 256), nn.ReLU()),
        #     *[nn.Sequential(nn.Linear(256, 256), nn.ReLU()) for _ in range(16)],
        #     nn.Sequential(nn.Linear(256, 1024), nn.ReLU()),
        #     *[nn.Sequential(nn.Linear(1024, 1024), nn.ReLU()) for _ in range(16)],
        #     nn.Sequential(nn.Linear(1024, 256), nn.ReLU()),
        #     *[nn.Sequential(nn.Linear(256, 256), nn.ReLU()) for _ in range(16)],
        #     nn.Sequential(nn.Linear(256, 64), nn.ReLU()),
        #     *[nn.Sequential(nn.Linear(64, 64), nn.ReLU()) for _ in range(8)],
        #     nn.Sequential(nn.Linear(64, self.hparams.features)),
        # )

        self.encoder = nn.Sequential(
            nn.Sequential(nn.Linear(self.hparams.features, 64), nn.ReLU()),
            *[nn.Sequential(nn.Linear(64, 64), nn.ReLU()) for _ in range(4)],
            pools.TopK(k=512, dim=1),
            nn.Sequential(nn.Linear(64, 256), nn.ReLU()),
            *[nn.Sequential(nn.Linear(256, 256), nn.ReLU()) for _ in range(6)],
            pools.Mean(dim=1),
            nn.Flatten(),
            *[nn.Sequential(nn.Linear(256, 256), nn.ReLU()) for _ in range(4)],
            nn.Sequential(nn.Linear(256, self.hparams.embeddings)),
        )

        self.flow = nn.Sequential(
            nn.Sequential(nn.Linear(self.hparams.features + self.hparams.conditions + self.hparams.embeddings, 64), nn.ReLU()),
            *[nn.Sequential(nn.Linear(64, 64), nn.ReLU()) for _ in range(4)],
            nn.Sequential(nn.Linear(64, 256), nn.ReLU()),
            *[nn.Sequential(nn.Linear(256, 256), nn.ReLU()) for _ in range(6)],
            nn.Sequential(nn.Linear(256, 64), nn.ReLU()),
            *[nn.Sequential(nn.Linear(64, 64), nn.ReLU()) for _ in range(4)],
            nn.Sequential(nn.Linear(64, self.hparams.features)),
        )

        # # initialize as random projection
        for layer in self.encoder.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

        nn.init.zeros_(self.encoder[-1][0].bias)

        # initialize as random drift
        for layer in self.flow.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

        nn.init.zeros_(self.flow[-1][0].bias)

    def compute_metrics(self, batch, batch_idx):
        sn0, sn1, ssn0, ssn1, ssnt, t, vstar = batch

        c = self.encoder(sn1)
        mmd_loss = L.mmd_loss(c, torch.randn_like(c), scales=torch.logspace(-20, 20, base=2, steps=41))

        v = self.velocity(ssnt, t, c)

        l1_loss = torch.mean(torch.abs(v - vstar))
        l2_loss = torch.mean(torch.square(v - vstar))

        loss = self.hparams.gamma * mmd_loss + (1 - self.hparams.gamma) * l2_loss

        return dict(
            loss=loss,
            l1_loss=l1_loss,
            l2_loss=l2_loss,
            mmd_loss=mmd_loss,
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

        t = 0.0
        dt = 1.0 / steps

        for i in range(steps):
            v = self.velocity(sn, t, c)
            sn = sn + v * dt
            t = t + dt

        return sn

    def train_dataloader(self):
        dl = super().train_dataloader()
        dl.collate_fn = cafm_collate
        return dl

    def val_dataloader(self):
        dl = super().val_dataloader()
        dl.collate_fn = cafm_collate
        return dl

    def test_dataloader(self):
        dl = super().test_dataloader()
        dl.collate_fn = cafm_collate
        return dl

    def configure_callbacks(self):
        callbacks = super().configure_callbacks()
        callbacks.append(SampleCallback())
        return callbacks
