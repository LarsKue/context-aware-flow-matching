
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

from .scatter import scatter, scatter_bp


def multiscatter(samples: np.ndarray, layout: (int, int) = None, **scatter_kwargs):
    sns.set_style("whitegrid", {"axes.grid": False})
    batch_size, set_size, *_ = samples.shape

    if layout is None:
        nrows = ncols = int(np.sqrt(batch_size))
    else:
        nrows, ncols = layout

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows), subplot_kw=dict(projection="3d"))

    for i, ax in enumerate(axes.flat):
        scatter(samples[i], ax=ax, **scatter_kwargs)

    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)

    return fig


def multiscatter_bp(samples: np.ndarray, layout: (int, int) = None, **render_kwargs):
    sns.set_style("whitegrid", {"axes.grid": False})
    batch_size, set_size, *_ = samples.shape

    if layout is None:
        nrows = ncols = int(np.sqrt(batch_size))
    else:
        nrows, ncols = layout

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))

    for ax in tqdm(axes.flat):
        scatter_bp(samples, ax=ax, **render_kwargs)

    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)

    return fig
