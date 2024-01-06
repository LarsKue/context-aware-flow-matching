
from lightning import Callback

import torch

from src import visualization as viz
from src.utils import temporary_seed


class SampleCallback(Callback):
    def __init__(self, num_plots: int = 4, num_samples: int = 2048, every_n_epochs: int = 2):
        self.num_plots = num_plots
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        with temporary_seed(42):
            samples = pl_module.sample((self.num_plots, self.num_samples)).cpu().numpy()

            fig = viz.multiscatter(samples)

            trainer.logger.experiment.add_figure("samples", fig, global_step=trainer.global_step)

        with temporary_seed(42):
            samples = torch.randperm(len(pl_module.val_data))[:self.num_plots]
            samples = torch.stack([pl_module.val_data[i][1] for i in samples])
            samples = samples.to(pl_module.device)

            samples = pl_module.reconstruct(samples).cpu().numpy()

            fig = viz.multiscatter(samples)

            trainer.logger.experiment.add_figure("reconstructions", fig, global_step=trainer.global_step)
