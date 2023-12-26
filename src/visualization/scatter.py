
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns

from .rainbow import Rainbow


def scatter(samples: np.ndarray, ax=plt.axes(projection="3d"), **scatter_kwargs):
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


def scatter_bp(samples: np.ndarray, ax=plt.gca(), **render_kwargs):
    """ This only works with blender-plot installed, which requires python==3.10.* """
    try:
        import blender_plot as bp
    except ImportError:
        raise ImportError(f"Using blender to render scatter plots requires blender-plot. "
                          f"Install the blender environment from blender.yaml to use this feature.")

    import sys

    sns.set_style("whitegrid", {"axes.grid": False})
    set_size, *_ = samples.shape

    render_kwargs.setdefault("resolution", (800, 800))

    scene = bp.DefaultScene()

    scene.scatter(samples)

    stdout = sys.stdout.write
    sys.stdout.write = lambda *args, **kwargs: None

    # this is super verbose, so we disable output
    scene.render("_.png")

    sys.stdout.write = stdout

    scene.clear()

    ax.imshow(plt.imread("_.png"))
    ax.set_axis_off()



