
import torch
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns

import src.utils as U

from .rainbow import Rainbow


def scatter(samples: np.ndarray, ax=None, **scatter_kwargs):
    if ax is None:
        ax = plt.axes(projection="3d")

    sns.set_style("whitegrid", {"axes.grid": False})

    scatter_kwargs.setdefault("s", 8)
    scatter_kwargs.setdefault("alpha", 1.0)
    scatter_kwargs.setdefault("marker", "o")
    scatter_kwargs.setdefault("lw", 0)

    cmap = Rainbow()

    c = cmap(samples)
    artist = ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c=c, **scatter_kwargs)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)

    ax.set_axis_off()

    return artist


def scatter_bp(samples: np.ndarray, ax=None, **render_kwargs):
    """ This only works with blender-plot installed, which requires python==3.10.* """
    if ax is None:
        ax = plt.axes()

    img = render_bp(samples, **render_kwargs)

    sns.set_style("whitegrid", {"axes.grid": False})

    artist = ax.imshow(img)
    ax.set_axis_off()

    return artist


def render_bp(samples: np.ndarray, name: str = "_.png", **render_kwargs):
    try:
        import blender_plot as bp
    except ImportError:
        raise ImportError(f"Using blender to render scatter plots requires blender-plot. "
                          f"Install the blender environment from blender.yaml to use this feature.")

    import sys

    set_size, *_ = samples.shape

    render_kwargs.setdefault("resolution", (800, 800))

    scene = bp.DefaultScene()

    angles = torch.Tensor([0, 0, -90])
    angles = angles.unsqueeze(0)
    angles = U.expand_dim(angles, 0, set_size)
    samples = U.rotate(torch.from_numpy(samples), angles).numpy()

    scene.scatter(samples)

    stdout = sys.stdout.write
    sys.stdout.write = lambda *args, **kwargs: None

    # this is super verbose, so we disable output
    scene.render(name, **render_kwargs)

    sys.stdout.write = stdout

    scene.clear()

    return plt.imread(name)
