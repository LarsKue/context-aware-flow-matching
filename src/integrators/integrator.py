
import torch
from torch import Tensor

from typing import Protocol

from tqdm import tqdm


class Integratable(Protocol):
    def __call__(self, x: Tensor, t: float, *conditions: Tensor) -> Tensor:
        raise NotImplementedError


class Integrator:
    """
    Base class to solve ordinary differential equations:

    dx = f(x, t, c) dt
    """
    def __init__(self, f: Integratable, t0: float, t1: float):
        self.f = f
        self.t0 = t0
        self.t1 = t1

    @torch.no_grad()
    def _dxdt(self, x: Tensor, t: float, *conditions: Tensor) -> (Tensor, float):
        raise NotImplementedError

    @torch.no_grad()
    def solve(self, x: Tensor, *conditions: Tensor, progress: bool = False) -> Tensor:
        x = x.clone()

        if progress:
            pbar = tqdm(total=self.t1 - self.t0, desc="Solving ODE", bar_format="{n:.3g}")

        t = self.t0
        while t < self.t1:
            dx, dt = self._dxdt(x, t, *conditions)

            x += dx
            t += dt

            if progress:
                # noinspection PyUnboundLocalVariable
                pbar.update(dt)

        return x


class FixedStepSizeIntegrator(Integrator):
    def __init__(self, f: Integratable, steps: int, t0: float = 0.0, t1: float = 1.0):
        super().__init__(f, t0, t1)
        self.steps = steps

    @torch.no_grad()
    def _dt(self):
        return (self.t1 - self.t0) / self.steps

    @torch.no_grad()
    def _dx(self, x: Tensor, t: float, *conditions: Tensor) -> Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def _dxdt(self, x: Tensor, t: float, *conditions: Tensor) -> (Tensor, float):
        return self._dx(x, t, *conditions), self._dt()
