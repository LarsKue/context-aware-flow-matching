
import torch
from torch import Tensor


from .integrator import FixedStepSizeIntegrator


class Euler(FixedStepSizeIntegrator):
    """
    Euler's method to solve ordinary differential equations:

    y_{n+1} = y_n + f(x_n, t_n, c) dt
    """
    @torch.no_grad()
    def _dx(self, x: Tensor, t: float, *conditions: Tensor) -> Tensor:
        return self.f(x, t, *conditions) * self._dt()
