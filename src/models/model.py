
import torch
import torch.nn as nn

from torch import Size, Tensor

from tqdm import tqdm, trange

from lightning_trainable import Trainable, TrainableHParams
from lightning_trainable.hparams import Range

import src.metrics as M
import src.utils as U

from src.callbacks import SampleCallback
from src.datasets import ContextAwareFlowMatchingDataset as CAFMDataset

from . import pools
from .skip_connection import ResidualBlock, SkipLinear


class ModelHParams(TrainableHParams):
    subset_size: int

    features: int = 3
    embeddings: int = 256

    gamma: Range(0.0, 1.0) = 0.5

    checkpoint_segments: int | None = None


class Model(Trainable):
    hparams: ModelHParams

    def __init__(self, hparams, *datasets):
        hparams = self.hparams_type(**hparams)
        datasets = [CAFMDataset(d, hparams.subset_size) for d in datasets]

        super().__init__(hparams, *datasets)

        # big networks

        # self.encoder = nn.Sequential(
        #     nn.Sequential(nn.Linear(self.hparams.features, 64), nn.SELU()),
        #
        #     ResidualBlock(64, 8, dropout=self.hparams.dropout),
        #
        #     SkipLinear(64),
        #     pools.TopK(k=1024, dim=1),
        #     SetNorm(64),
        #     nn.Sequential(nn.Linear(64, 128), nn.SELU()),
        #
        #     ResidualBlock(128, 8, dropout=self.hparams.dropout),
        #
        #     SkipLinear(128),
        #     pools.TopK(k=512, dim=1),
        #     SetNorm(128),
        #
        #     ResidualBlock(128, 8, dropout=self.hparams.dropout),
        #
        #     SkipLinear(128),
        #     pools.TopK(k=256, dim=1),
        #     SetNorm(128),
        #     nn.Sequential(nn.Linear(128, 256), nn.SELU()),
        #
        #     ResidualBlock(256, 8, dropout=self.hparams.dropout),
        #
        #     SkipLinear(256),
        #     pools.TopK(k=128, dim=1),
        #     SetNorm(256),
        #
        #     ResidualBlock(256, 8, dropout=self.hparams.dropout),
        #
        #     SkipLinear(256),
        #     pools.TopK(k=64, dim=1),
        #     SetNorm(256),
        #     nn.Sequential(nn.Linear(256, 512), nn.SELU()),
        #
        #     ResidualBlock(512, 8, dropout=self.hparams.dropout),
        #
        #     SkipLinear(512),
        #     pools.TopK(k=32, dim=1),
        #     SetNorm(512),
        #
        #     ResidualBlock(512, 8, dropout=self.hparams.dropout),
        #
        #     SkipLinear(512),
        #     pools.TopK(k=16, dim=1),
        #     SetNorm(512),
        #
        #     ResidualBlock(512, 8, dropout=self.hparams.dropout),
        #
        #     SkipLinear(512),
        #     pools.TopK(k=8, dim=1),
        #     SetNorm(512),
        #     nn.Sequential(nn.Linear(512, 1024), nn.SELU()),
        #
        #     ResidualBlock(1024, 4, dropout=self.hparams.dropout),
        #
        #     SkipLinear(1024),
        #     pools.Mean(dim=1),
        #     nn.Flatten(),
        #
        #     nn.BatchNorm1d(1024),
        #
        #     ResidualBlock(1024, 4, dropout=self.hparams.dropout),
        #
        #     nn.Sequential(nn.Linear(1024, 512), nn.SELU()),
        #
        #     nn.BatchNorm1d(512),
        #
        #     ResidualBlock(512, 4, dropout=self.hparams.dropout),
        #
        #     nn.Linear(512, self.hparams.embeddings),
        # )
        #
        # self.flow = nn.Sequential(
        #     nn.Sequential(nn.Linear(self.hparams.features + 1 + self.hparams.embeddings, 512), nn.SELU()),
        #     ResidualBlock(512, 8, dropout=self.hparams.dropout),
        #
        #     nn.LayerNorm(512),
        #
        #     nn.Sequential(nn.Linear(512, 1024), nn.SELU()),
        #     ResidualBlock(1024, 8, dropout=self.hparams.dropout),
        #
        #     nn.LayerNorm(1024),
        #
        #     nn.Sequential(nn.Linear(1024, 512), nn.SELU()),
        #     ResidualBlock(512, 8, dropout=self.hparams.dropout),
        #
        #     nn.LayerNorm(512),
        #
        #     nn.Sequential(nn.Linear(512, 256), nn.SELU()),
        #     ResidualBlock(256, 8, dropout=self.hparams.dropout),
        #
        #     nn.LayerNorm(256),
        #
        #     nn.Sequential(nn.Linear(256, 128), nn.SELU()),
        #     ResidualBlock(128, 8, dropout=self.hparams.dropout),
        #
        #     nn.LayerNorm(128),
        #
        #     nn.Sequential(nn.Linear(128, 64), nn.SELU()),
        #     ResidualBlock(64, 8, dropout=self.hparams.dropout),
        #
        #     nn.Linear(64, self.hparams.features),
        # )

        # medium networks (~15M params)

        self.encoder = nn.Sequential(
            nn.Sequential(nn.Linear(self.hparams.features, 64), nn.SELU()),
            ResidualBlock(64, 4),

            SkipLinear(64),
            pools.TopK(k=1024, dim=1),
            nn.LayerNorm(64),

            ResidualBlock(64, 4),

            SkipLinear(64),
            pools.TopK(k=512, dim=1),
            nn.LayerNorm(64),

            nn.Sequential(nn.Linear(64, 128), nn.SELU()),
            ResidualBlock(128, 4),

            SkipLinear(128),
            pools.TopK(k=256, dim=1),
            nn.LayerNorm(128),

            ResidualBlock(128, 4),

            SkipLinear(128),
            pools.TopK(k=128, dim=1),
            nn.LayerNorm(128),

            nn.Sequential(nn.Linear(128, 256), nn.SELU()),
            ResidualBlock(256, 4),

            SkipLinear(256),
            pools.TopK(k=64, dim=1),
            nn.LayerNorm(256),

            ResidualBlock(256, 4),

            SkipLinear(256),
            pools.TopK(k=32, dim=1),
            nn.LayerNorm(256),

            nn.Sequential(nn.Linear(256, 512), nn.SELU()),
            ResidualBlock(512, 4),

            SkipLinear(512),
            pools.TopK(k=16, dim=1),
            nn.LayerNorm(512),

            ResidualBlock(512, 4),

            SkipLinear(512),
            pools.TopK(k=8, dim=1),
            nn.LayerNorm(512),

            nn.Sequential(nn.Linear(512, 1024), nn.SELU()),
            ResidualBlock(1024, 8),

            SkipLinear(1024),
            pools.Mean(dim=1),
            nn.Flatten(),

            ResidualBlock(1024, 8),

            nn.Sequential(nn.Linear(1024, 512), nn.SELU()),
            ResidualBlock(512, 4),

            nn.Linear(512, self.hparams.embeddings),
        )

        self.flow = nn.Sequential(
            nn.Sequential(nn.Linear(self.hparams.features + self.hparams.embeddings + 1, 512), nn.SELU()),
            ResidualBlock(512, 8),

            nn.Sequential(nn.Linear(512, 1024), nn.SELU()),
            ResidualBlock(1024, 8),

            nn.Sequential(nn.Linear(1024, 512), nn.SELU()),
            ResidualBlock(512, 8),

            nn.Sequential(nn.Linear(512, 256), nn.SELU()),
            ResidualBlock(256, 4),

            nn.Sequential(nn.Linear(256, 128), nn.SELU()),
            ResidualBlock(128, 4),

            nn.Sequential(nn.Linear(128, 64), nn.SELU()),
            ResidualBlock(64, 2),

            nn.Linear(64, self.hparams.features),
        )

        # small networks for testing

        # self.encoder = nn.Sequential(
        #     nn.Sequential(nn.Linear(self.hparams.features, 64), nn.SELU()),
        #     *[nn.Sequential(SkipLinear(64), nn.SELU()) for _ in range(2)],
        #
        #     SkipLinear(64),
        #     pools.TopK(k=512, dim=1),
        #
        #     nn.Sequential(nn.Linear(64, 256), nn.SELU()),
        #     *[nn.Sequential(SkipLinear(256), nn.SELU()) for _ in range(2)],
        #
        #     SkipLinear(256),
        #     pools.TopK(k=128, dim=1),
        #
        #     nn.Sequential(nn.Linear(256, 512), nn.SELU()),
        #     *[nn.Sequential(SkipLinear(512), nn.SELU()) for _ in range(2)],
        #
        #     SkipLinear(512),
        #     pools.Mean(dim=1),
        #     nn.Flatten(),
        #
        #     *[nn.Sequential(SkipLinear(512), nn.SELU()) for _ in range(4)],
        #     nn.Linear(512, self.hparams.embeddings),
        # )
        #
        # self.flow = nn.Sequential(
        #     nn.Sequential(nn.Linear(self.hparams.features + 1 + self.hparams.embeddings, 512), nn.SELU()),
        #     *[nn.Sequential(SkipLinear(512), nn.SELU()) for _ in range(2)],
        #
        #     nn.Sequential(nn.Linear(512, 256), nn.SELU()),
        #     *[nn.Sequential(SkipLinear(256), nn.SELU()) for _ in range(2)],
        #
        #     nn.Sequential(nn.Linear(256, 128), nn.SELU()),
        #     *[nn.Sequential(SkipLinear(128), nn.SELU()) for _ in range(2)],
        #
        #     nn.Sequential(nn.Linear(128, 64), nn.SELU()),
        #     *[nn.Sequential(SkipLinear(64), nn.SELU()) for _ in range(2)],
        #
        #     nn.Linear(64, self.hparams.features),
        # )

        for layer in self.encoder.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=1.0 / layer.out_features)
                nn.init.zeros_(layer.bias)

        for layer in self.flow.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=1.0 / layer.out_features)
                nn.init.zeros_(layer.bias)

        if self.hparams.checkpoint_segments is not None:
            segments = self.hparams.checkpoint_segments
            self.encoder = U.CheckpointedSequential(*self.encoder, segments=segments)
            self.flow = U.CheckpointedSequential(*self.flow, segments=segments)

    def compute_metrics(self, batch, batch_idx):
        sn0, sn1, ssn0, ssn1, ssnt, t, vstar = batch

        c = self.encoder(sn1)
        mmd = M.maximum_mean_discrepancy(c, torch.randn_like(c), scales=torch.logspace(-20, 20, base=2, steps=41))

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
    def sample(self, sample_shape: Size = (1,), steps: int = 100, progress: bool = False) -> Tensor:
        sn = torch.randn(sample_shape[0], sample_shape[1], self.hparams.features, device=self.device)
        c = torch.randn(sample_shape[0], self.hparams.embeddings, device=self.device)

        return self.sample_from(sn, c, steps=steps, progress=progress)

    @torch.no_grad()
    def sample_from(self, noise: Tensor, condition: Tensor, steps: int = 100, progress: bool = False) -> Tensor:
        t = 0.0
        dt = 1.0 / steps

        if progress:
            steps = trange(steps)
        else:
            steps = range(steps)

        for step in steps:
            velocity = self.velocity(noise, t, condition)
            noise = noise + velocity * dt
            t = t + dt

        return noise

    def configure_callbacks(self):
        callbacks = super().configure_callbacks()
        callbacks.append(SampleCallback())
        return callbacks
