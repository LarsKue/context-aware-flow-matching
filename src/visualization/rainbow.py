
from matplotlib.colors import Colormap

import numpy as np


class Rainbow(Colormap):
    def __init__(self):
        super().__init__("rainbow")

    def __call__(self, X, alpha=None, bytes=False):
        x = X[:, 0:1]
        y = X[:, 1:2]
        z = X[:, 2:3]

        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        z = (z - np.min(z)) / (np.max(z) - np.min(z))

        red = np.array([[1.0, 0.0, 0.0]])
        green = np.array([[0.0, 1.0, 0.0]])
        blue = np.array([[0.0, 0.0, 1.0]])
        cyan = np.array([[0.0, 1.0, 1.0]])
        magenta = np.array([[1.0, 0.0, 1.0]])
        yellow = np.array([[1.0, 1.0, 0.0]])

        c1 = x * red + (1 - x) * cyan
        c2 = y * green + (1 - y) * magenta
        c3 = z * blue + (1 - z) * yellow

        rgb = np.stack([c1[:, 0], c2[:, 1], c3[:, 2]], axis=1)

        if alpha is None:
            alpha = np.ones((rgb.shape[0], 1))
        elif isinstance(alpha, float):
            alpha = np.ones((rgb.shape[0], 1)) * alpha

        rgba = np.concatenate([rgb, alpha], axis=1)

        return rgba
