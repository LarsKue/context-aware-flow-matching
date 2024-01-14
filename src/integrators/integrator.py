
import torch
from torch import Tensor

from typing import Protocol, Callable

from tqdm import tqdm


Integratable = Callable[[Tensor, float, ...], Tensor]


class Integrator:
    """
    Base class to solve ordinary differential equations:

    dx = f(x, t, c) dt
    """

    BAR_FORMAT: str = "{l_bar}{bar}| {n:.3g}/{total:.3g} [{elapsed}<{remaining}]"

    def __init__(self, f: Integratable, t0: float, t1: float):
        self.f = f
        self.t0 = t0
        self.t1 = t1

    @torch.no_grad()
    def _dxdt(self, x: Tensor, t: float, *args, **kwargs) -> (Tensor, float):
        raise NotImplementedError

    @torch.no_grad()
    def trajectory(self, x: Tensor, *args, progress: bool = False, **kwargs) -> Tensor:
        x = x.clone().detach()

        if progress:
            pbar = tqdm(total=self.t1 - self.t0, desc="Computing trajectory", bar_format=self.BAR_FORMAT)

        t = self.t0
        trajectory = [x.clone()]
        while t < self.t1:
            dx, dt = self._dxdt(x, t, *args, **kwargs)

            x += dx
            t += dt

            trajectory.append(x.clone())

            if progress:
                # noinspection PyUnboundLocalVariable
                pbar.update(dt)

        return torch.stack(trajectory)

    @torch.no_grad()
    def solve(self, x: Tensor, *args, progress: bool = False, **kwargs) -> Tensor:
        x = x.clone().detach()

        if progress:
            pbar = tqdm(total=self.t1 - self.t0, desc="Solving ODE", bar_format=self.BAR_FORMAT)

        t = self.t0
        while t < self.t1:
            dx, dt = self._dxdt(x, t, *args, **kwargs)

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
    def _dx(self, x: Tensor, t: float, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def _dxdt(self, x: Tensor, t: float, *args, **kwargs) -> (Tensor, float):
        return self._dx(x, t, *args, **kwargs), self._dt()
