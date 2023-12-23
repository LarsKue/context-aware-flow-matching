from functools import wraps

import torch
import numpy as np
import random


class temporary_seed:
    """
    context manager or decorator to use a temporary seed for machine-learning relevant libraries
    Includes:
        - PyTorch CPU
        - PyTorch CUDA
        - Numpy
        - Python

    Usage:

    As a context manager:
    >>> with temporary_seed(42):
    >>>     # do something

    As a decorator:
    >>> @temporary_seed(42)
    >>> def function():
    >>>     # do something

    """
    def __init__(self, seed: int):
        self.seed = seed
        self.states = {}

    def __call__(self, function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with self:
                return function(*args, **kwargs)

        return wrapped

    def __enter__(self):
        self.states = {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(),
            "python": random.getstate()
        }

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        random.setstate(self.states["python"])
        np.random.set_state(self.states["numpy"])
        torch.cuda.set_rng_state_all(self.states["cuda"])
        torch.set_rng_state(self.states["cpu"])
