
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap

import numpy as np
import seaborn as sns

from .rainbow import Rainbow


def multiscatter(samples: np.ndarray):
    sns.set_style("whitegrid", {"axes.grid": False})
    batch_size, set_size, *_ = samples.shape
    nrows = ncols = int(np.sqrt(batch_size))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw=dict(projection="3d"))

    cmap = Rainbow()

    for i, ax in enumerate(axes.flat):
        c = cmap(samples[i])
        ax.scatter(samples[i, :, 0], samples[i, :, 1], samples[i, :, 2], s=4, alpha=1.0, c=c)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-3, 3)

        ax.set_axis_off()


    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.01, hspace=0.01)

    return fig
