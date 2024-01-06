
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

from src.integrators import Euler, RK45

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

            ResidualBlock(1024, 12),

            nn.Linear(1024, self.hparams.embeddings),
        )

        self.flow = nn.Sequential(
            nn.Sequential(nn.Linear(self.hparams.features + self.hparams.embeddings + 1, 1024), nn.SELU()),
            ResidualBlock(1024, 16),

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
            self.encoder = U.CheckpointedSequential.from_nested(self.encoder, segments=segments)
            self.flow = U.CheckpointedSequential.from_nested(self.flow, segments=segments)

    def compute_metrics(self, batch, batch_idx):
        sn0, sn1, ssn0, ssn1, ssnt, t, vstar = batch

        embedding = self.embed(sn1)
        mmd = M.maximum_mean_discrepancy(embedding, torch.randn_like(embedding), scales=torch.logspace(-20, 20, base=2, steps=41))

        v = self.velocity(ssnt, t, embedding)

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

    def embed(self, samples: Tensor) -> Tensor:
        batch_size, set_size, _ = samples.shape
        return self.encoder(samples)

    def velocity(self, x: Tensor, t: Tensor | float, embedding: Tensor) -> Tensor:
        batch_size, set_size, _ = x.shape

        if isinstance(t, float):
            t = torch.full((batch_size,), fill_value=t, device=self.device)

        t = U.unsqueeze_right(t, x.dim() - t.dim())
        t = U.expand_dim(t, 1, set_size)

        embedding = embedding.unsqueeze(1)
        embedding = U.expand_dim(embedding, 1, set_size)

        sntc = torch.cat([x, t, embedding], dim=2)

        return self.flow(sntc)

    @torch.no_grad()
    def sample(self, sample_shape: Size = (1, 128), integrator: str = "euler", steps: int = 100, progress: bool = False) -> Tensor:
        batch_size, set_size = sample_shape
        noise = self._sample_noise(Size((batch_size, set_size)))
        embedding = self._sample_embedding(Size((batch_size,)))

        return self.sample_from(noise, embedding, integrator=integrator, steps=steps, progress=progress)

    @torch.no_grad()
    def _sample_noise(self, sample_shape: Size = (1, 128)) -> Tensor:
        batch_size, set_size = sample_shape
        return torch.randn(batch_size, set_size, self.hparams.features, device=self.device)

    @torch.no_grad()
    def _sample_embedding(self, sample_shape: Size = (1,)) -> Tensor:
        batch_size, = sample_shape
        return torch.randn(batch_size, self.hparams.embeddings, device=self.device)

    @torch.no_grad()
    def sample_from(self, noise: Tensor, embedding: Tensor, integrator: str = "euler", steps: int = 100, progress: bool = False) -> Tensor:
        match integrator:
            case "euler":
                integrator = Euler(self.velocity, steps=steps)
            case "rk45":
                integrator = RK45(self.velocity, steps=steps)
            case _:
                raise ValueError(f"Unknown integrator: {integrator}")

        return integrator.solve(noise, embedding, progress=progress)

    @torch.no_grad()
    def reconstruct(self, samples: Tensor, integrator: str = "euler", steps: int = 100, progress: bool = False) -> Tensor:
        batch_size, set_size, _ = samples.shape
        noise = self._sample_noise(Size((batch_size, set_size)))
        embedding = self.embed(samples)

        return self.sample_from(noise, embedding, integrator=integrator, steps=steps, progress=progress)

    def configure_callbacks(self):
        callbacks = super().configure_callbacks()
        callbacks.append(SampleCallback())
        return callbacks
