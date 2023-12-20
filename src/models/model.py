
import torch
import torch.nn as nn

from torch import Tensor

from . import pools


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # 3d point clouds
        features = 3
        # time
        conditions = 1
        # encoder latent space
        embeddings = 256

        self.encoder = nn.Sequential(
            nn.Sequential(nn.Linear(features, 64), nn.ReLU()),
            *[nn.Sequential(nn.Linear(64, 64), nn.ReLU()) for _ in range(8)],
            pools.TopK(k=512, dim=1),
            nn.Sequential(nn.Linear(64, 256), nn.ReLU()),
            *[nn.Sequential(nn.Linear(256, 256), nn.ReLU()) for _ in range(16)],
            pools.TopK(k=64, dim=1),
            nn.Sequential(nn.Linear(256, 1024), nn.ReLU()),
            *[nn.Sequential(nn.Linear(1024, 1024), nn.ReLU()) for _ in range(16)],
            pools.Mean(dim=1),
            *[nn.Sequential(nn.Linear(1024, 1024), nn.ReLU()) for _ in range(8)],
            nn.Linear(1024, embeddings),
        )

        # diamond shape flow matching model
        self.flow = nn.Sequential(
            nn.Sequential(nn.Linear(features + conditions + embeddings, 64), nn.ReLU()),
            *[nn.Sequential(nn.Linear(64, 64), nn.ReLU()) for _ in range(8)],
            nn.Sequential(nn.Linear(64, 256), nn.ReLU()),
            *[nn.Sequential(nn.Linear(256, 256), nn.ReLU()) for _ in range(16)],
            nn.Sequential(nn.Linear(256, 1024), nn.ReLU()),
            *[nn.Sequential(nn.Linear(1024, 1024), nn.ReLU()) for _ in range(16)],
            nn.Sequential(nn.Linear(1024, 256), nn.ReLU()),
            *[nn.Sequential(nn.Linear(256, 256), nn.ReLU()) for _ in range(16)],
            nn.Sequential(nn.Linear(256, 64), nn.ReLU()),
            *[nn.Sequential(nn.Linear(64, 64), nn.ReLU()) for _ in range(8)],
            nn.Sequential(nn.Linear(64, 3)),
        )

        # initialize as random projection
        for layer in self.encoder.children():
            if isinstance(layer, nn.Linear):
                print("Initialising linear layer in encoder")
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # initialize as random drift
        for layer in self.flow.children():
            if isinstance(layer, nn.Linear):
                print("Initialising linear layer in flow")
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def velocity(self, x: Tensor, t: Tensor, c: Tensor) -> Tensor:
        xtc = torch.cat([x, t, c], dim=1)

        return self.flow(xtc)



    def training_step(self, batch, batch_idx):
        xtc, vstar = batch

        v = self.flow(xtc)

        mse = F.mse_loss(v, vstar)

        return dict(
            loss=mse,
            mse=mse,
        )

        x0, x1 = batch
        batch_size, set_size, *_ = x0.shape

        t = torch.rand(batch_size, device=self.device)
        xt = t * x1 + (1 - t) * x0

        embedding = self.encoder(x0)

        xtc = torch.cat([xt, t, embedding], dim=2)

        velocity = self.flow(xtc)
