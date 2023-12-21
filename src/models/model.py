
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import DataLoader

from lightning_trainable import Trainable, TrainableHParams

import src.losses as L
import src.utils as U

from src.dataloaders import FlowMatchingDataLoader

from . import pools


class ModelHParams(TrainableHParams):
    features: int = 3
    conditions: int = 1
    embeddings: int = 256


class Model(Trainable):
    hparams: ModelHParams

    def __init__(self, hparams, *datasets):
        super().__init__(hparams, *datasets)

        self.encoder = nn.Sequential(
            nn.Sequential(nn.Linear(self.hparams.features, 64), nn.ReLU()),
            *[nn.Sequential(nn.Linear(64, 64), nn.ReLU()) for _ in range(8)],
            pools.TopK(k=512, dim=1),
            nn.Sequential(nn.Linear(64, 256), nn.ReLU()),
            *[nn.Sequential(nn.Linear(256, 256), nn.ReLU()) for _ in range(16)],
            pools.TopK(k=64, dim=1),
            nn.Sequential(nn.Linear(256, 1024), nn.ReLU()),
            *[nn.Sequential(nn.Linear(1024, 1024), nn.ReLU()) for _ in range(16)],
            pools.Mean(dim=1),
            nn.Flatten(),
            *[nn.Sequential(nn.Linear(1024, 1024), nn.ReLU()) for _ in range(8)],
            nn.Sequential(nn.Linear(1024, self.hparams.embeddings)),
        )

        # diamond shape flow matching model
        self.flow = nn.Sequential(
            nn.Sequential(nn.Linear(self.hparams.features + self.hparams.conditions + self.hparams.embeddings, 64), nn.ReLU()),
            *[nn.Sequential(nn.Linear(64, 64), nn.ReLU()) for _ in range(8)],
            nn.Sequential(nn.Linear(64, 256), nn.ReLU()),
            *[nn.Sequential(nn.Linear(256, 256), nn.ReLU()) for _ in range(16)],
            nn.Sequential(nn.Linear(256, 1024), nn.ReLU()),
            *[nn.Sequential(nn.Linear(1024, 1024), nn.ReLU()) for _ in range(16)],
            nn.Sequential(nn.Linear(1024, 256), nn.ReLU()),
            *[nn.Sequential(nn.Linear(256, 256), nn.ReLU()) for _ in range(16)],
            nn.Sequential(nn.Linear(256, 64), nn.ReLU()),
            *[nn.Sequential(nn.Linear(64, 64), nn.ReLU()) for _ in range(8)],
            nn.Sequential(nn.Linear(64, self.hparams.features)),
        )

        # initialize as random projection
        for layer in self.encoder.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        # initialize as random drift
        for layer in self.flow.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def velocity(self, x: Tensor, t: Tensor, c: Tensor) -> Tensor:
        xtc = torch.cat([x, t, c], dim=1)

        return self.flow(xtc)

    def compute_metrics(self, batch, batch_idx):
        xt, x0, x1, t, vstar = batch

        batch_size, set_size, *_ = x0.shape

        c = self.encoder(x0)

        mmd = L.mmd_loss(c, torch.randn_like(c), reduction="none")
        mmd_mean = mmd.mean(0)

        c = c.unsqueeze(1)
        c = U.expand_dim(c, 1, set_size)
        t = U.unsqueeze_right(t, xt.dim() - 1)
        t = U.expand_dim(t, 1, set_size)

        xtc = torch.cat([xt, t, c], dim=2)
        v = self.flow(xtc)

        mse = U.mean_except(torch.square(v - vstar), 0)
        mse_mean = mse.mean(0)

        loss = self.hparams.gamma * mmd + (1 - self.hparams.gamma) * mse
        loss_mean = loss.mean(0)

        return dict(
            loss=loss_mean,
            mse=mse_mean,
            mmd=mmd_mean,
        )

    def _convert_dataloader(self, dataloader: DataLoader) -> DataLoader:
        # we don't want to rewrite all the customizing logic, so just cast to our custom class
        # this works, because the class is a subclass of DataLoader and does not introduce any new
        # attributes or methods
        dataloader.__class__ = FlowMatchingDataLoader

        return dataloader

    def train_dataloader(self) -> DataLoader | list[DataLoader]:
        return self._convert_dataloader(super().train_dataloader())

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        return self._convert_dataloader(super().val_dataloader())

    def test_dataloader(self) -> DataLoader | list[DataLoader]:
        return self._convert_dataloader(super().test_dataloader())
