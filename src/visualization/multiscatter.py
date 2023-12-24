
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns

from tqdm import tqdm

from .rainbow import Rainbow



def multiscatter(samples: np.ndarray, **scatter_kwargs):
    sns.set_style("whitegrid", {"axes.grid": False})
    batch_size, set_size, *_ = samples.shape
    nrows = ncols = int(np.sqrt(batch_size))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw=dict(projection="3d"))

    cmap = Rainbow()

    scatter_kwargs.setdefault("s", 8)
    scatter_kwargs.setdefault("alpha", 1.0)
    scatter_kwargs.setdefault("marker", "o")
    scatter_kwargs.setdefault("lw", 0)

    for i, ax in enumerate(axes.flat):
        c = cmap(samples[i])
        ax.scatter(samples[i, :, 0], samples[i, :, 1], samples[i, :, 2], c=c, **scatter_kwargs)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)

        ax.set_axis_off()

    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)

    return fig


def multiscatter_bp(samples: np.ndarray, **render_kwargs):
    """ This only works with blender-plot installed, which requires python==3.10.* """
    import blender_plot as bp
    import sys

    sns.set_style("whitegrid", {"axes.grid": False})
    batch_size, set_size, *_ = samples.shape
    nrows = ncols = int(np.sqrt(batch_size))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

    render_kwargs.setdefault("resolution", (800, 800))

    for i, ax in tqdm(list(enumerate(axes.flat))):
        scene = bp.DefaultScene()

        scene.scatter(samples[i])

        stdout = sys.stdout.write
        sys.stdout.write = lambda *args, **kwargs: None

        # this is super verbose, so we disable output
        scene.render("_.png")

        sys.stdout.write = stdout

        scene.clear()

        ax.imshow(plt.imread("_.png"))
        ax.set_axis_off()

    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)

    return fig
