
from lightning import Callback

import torch

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class SampleCallback(Callback):
    def __init__(self, num_plots: int = 4, num_samples: int = 512, every_n_epochs: int = 1):
        self.num_plots = num_plots
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        sns.set_style("whitegrid", {"axes.grid": False})

        nrows, ncols = int(np.ceil(np.sqrt(self.num_plots))), int(np.ceil(np.sqrt(self.num_plots)))
        fig = plt.figure(figsize=(8, 8))

        axes = []

        for row in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(nrows, ncols, row * ncols + col + 1, projection="3d")
                axes.append(ax)

        samples = pl_module.sample((self.num_plots, self.num_samples)).cpu().numpy()

        for i in range(self.num_plots):
            axes[i].scatter(samples[i, :, 0], samples[i, :, 1], marker="o", linewidth=0, alpha=0.8)
            axes[i].set_xlim(-3, 3)
            axes[i].set_ylim(-3, 3)
            axes[i].set_zlim(-3, 3)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_zticks([])
            axes[i].set_xlabel("x")
            axes[i].set_ylabel("y")
            axes[i].set_zlabel("z")

        trainer.logger.experiment.add_figure("samples", fig, global_step=trainer.global_step)

