
import torch
from torch import Tensor

from .integrator import FixedStepSizeIntegrator


class RK45(FixedStepSizeIntegrator):
    """
    Runge-Kutta 4th order method to solve ordinary differential equations:

    y_{n+1} = y_n + (k_1 + 2 k_2 + 2 k_3 + k_4) dt / 6

        where

        k_1 = f(x_n, t_n, c)
        k_2 = f(x_n + dt * 0.5 * k_1, t_n + 0.5 * dt, c)
        k_3 = f(x_n + dt * 0.5 * k_2, t_n + 0.5 * dt, c)
        k_4 = f(x_n + dt * k_3, t_n + dt, c)
    """

    @torch.no_grad()
    def _dx(self, x: Tensor, t: float, *args, **kwargs) -> Tensor:
        dt = self._dt()

        k1 = self.f(x, t, *args, **kwargs)
        k2 = self.f(x + dt * 0.5 * k1, t + 0.5 * dt, *args, **kwargs)
        k3 = self.f(x + dt * 0.5 * k2, t + 0.5 * dt, *args, **kwargs)
        k4 = self.f(x + dt * k3, t + dt, *args, **kwargs)

        return (k1 + 2.0 * k2 + 2.0 * k3 + k4) * dt / 6.0
