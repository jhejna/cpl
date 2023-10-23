import collections
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

LOG_FILE_NAME = "log.csv"

sns.set_context(context="paper", font_scale=0.68)
sns.set_style("white", {"font.family": "serif"})


def moving_avg(x, y, window_size: int = 1):
    if window_size == 1:
        return x, y
    moving_avg_y = np.convolve(y, np.ones(window_size) / window_size, "valid")
    return x[-len(moving_avg_y) :], moving_avg_y


def plot_run(
    paths: List[str],
    name: str,
    ax=None,
    x_key: str = "step",
    y_keys: Optional[List[str]] = None,
    window_size: int = 1,
    max_x_value: Optional[int] = None,
    **kwargs,
) -> None:
    for path in paths:
        assert LOG_FILE_NAME in os.listdir(path), (
            "Did not find log file for " + path + ", found " + " ".join(os.listdir(path))
        )
    y_keys = ["validation/loss"] if y_keys is None else y_keys
    for y_key in y_keys:
        xs, ys = [], []
        label = name + " " + y_key if len(y_keys) > 1 else name
        for path in paths:
            if LOG_FILE_NAME not in os.listdir(path):
                print("skipping", path)
                continue
            df = pd.read_csv(os.path.join(path, LOG_FILE_NAME))
            if y_key not in df:
                print("[research] WARNING: y_key was not in run, skipping", path)
                continue
            x, y = df[x_key].to_numpy(), df[y_key].to_numpy()
            assert len(x) == len(y)
            if max_x_value is not None:
                y = y[x <= max_x_value]  # need to set y value first
                x = x[x <= max_x_value]
            xs.append(x)
            ys.append(y)

        if len(ys) == 0:
            print("[research], WARNING: had no runs for y_key", y_key, "skipping.")
            continue

        # Determine if we should trim runs to make sure it works for table printing
        force_table = False
        if force_table:
            min_x = max((x[0] for x in xs))
            xs, ys = zip(*[(x[np.where(x == min_x)[0][0] :], y[np.where(x == min_x)[0][0] :]) for x, y in zip(xs, ys)])
            min_len = min((len(x) for x in xs))
            xs = [x[:min_len] for x in xs]
            ys = [y[:min_len] for y in ys]

        # Run smoothing
        xs, ys = zip(*[moving_avg(x, y, window_size=window_size) for x, y in zip(xs, ys)])

        # Compute the table statistics for printing if all runs have finished.
        if all(len(xs[0]) == len(x) for x in xs):
            xs = np.stack(xs, axis=0)  # Shape (Seeds, Length)
            ys = np.stack(ys, axis=0)
            assert np.all(xs[0:1] == xs), "X Values must be the same"
            means = np.mean(ys, axis=0)
            stds = np.std(ys, axis=0)
            max_mean = np.max(means)
            max_std = stds[np.argmax(means)]
            print("Max: {:.1f} \\stdv{{{:.1f}}}".format(max_mean * 100, max_std * 100), label)
            print("{:.1f} \\stdv{{{:.1f}}}".format(means[-1] * 100, stds[-1] * 100))
            print("Last: {:.1f} \\stdv{{{:.1f}}}".format(means[-1] * 100, stds[-1] * 100), label)
            xs = xs.flatten()
            ys = ys.flatten()
        else:
            xs = np.concatenate(xs, axis=0)
            ys = np.concatenate(ys, axis=0)

        plot_df = pd.DataFrame({x_key: xs, y_key: ys})

        errorbar = "sd" if len(paths) > 0 else None
        sns.lineplot(ax=ax, x=x_key, y=y_key, data=plot_df, sort=True, errorbar=errorbar, label=label, **kwargs)


def create_plot(
    paths: List[Union[str, float]],
    labels: List[str],
    ax=None,
    title: Optional[str] = None,
    color_map: Optional[Dict] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    xticks: Optional[List] = None,
    yticks: Optional[List] = None,
    xticklabels: Optional[List[str]] = None,
    yticklabels: Optional[List[str]] = None,
    vline: Optional[float] = None,
    **kwargs,
):
    assert len(labels) == len(labels), "The length of paths must the same as the length of labels"
    ax = plt.gca() if ax is None else ax
    if vline is not None:
        ax.axvline(x=vline, color="black", linestyle="--")

    # Setup the color map
    if color_map is None:
        color_map = collections.defaultdict(lambda: None)
    for k in color_map.keys():
        if isinstance(color_map[k], int):
            color_map[k] = sns.color_palette()[color_map[k]]
        elif isinstance(color_map[k], (tuple, list)):
            assert len(color_map[k]) == 3
            if isinstance(color_map[k][0], int):
                color_map[k] = [v / 255.0 for v in color_map[k]]

    # Construct the plots
    for path, label in zip(paths, labels):
        if isinstance(path, float):
            # if its just a float, add it as a horizontile line.
            ax.axhline(y=path, color=color_map[label], label=label)
            continue
        elif LOG_FILE_NAME not in os.listdir(path):
            # If we have multiple seeds in the same directory, add them to a list.
            run_paths = [os.path.join(path, run) for run in os.listdir(path) if run.startswith("seed")]
        else:
            run_paths = [path]
        plot_run(run_paths, label, ax=ax, color=color_map[label], **kwargs)

    ax.set_title(title, pad=1)

    # Set tick parameters
    ax.tick_params(axis="y", pad=-2, labelsize=5)
    ax.tick_params(axis="x", pad=-2, labelsize=5)
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticks is not None:
        ax.set_yticks(yticks)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)

    # Set label parameters
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel, labelpad=0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    sns.despine(ax=ax)


def plot_from_config(config_path: str) -> None:
    """
    --- Configuration design for plot files ---
    title: null
    kwargs:
        xlabel: etc.
        ylabel: etc.
    color_map:
        method_1: idx
        method_2: idx

    grid_shape: (rows, cols)
    fig_size: (6, 3) etc. or null
    legend_pos: first
    use_subplot_titles: true

    plots:
        title_1:
            methods:
                method_1: path
                method_2: path
            kwargs:
            image: image path if we want to add an image

        title_2:
            methods:
                method_1: path
                method_2: path
            config:
            image: image path
    """
    import yaml

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    grid_shape = config["grid_shape"]
    rect = [0, 0, 1, 1]

    # Note that grid shape is given as (rows, cols)
    assert len(config["plots"]) <= grid_shape[0] * grid_shape[1]
    figsize = (2 * grid_shape[1], grid_shape[0]) if config.get("fig_size") is None else config.get("fig_size")

    legend_pos = config.get("legend_pos")
    assert legend_pos in {"first", "last", "bottom", "all", None}
    if legend_pos == "first":
        legend_index = 0
    elif legend_pos == "last":
        legend_index = len(config["plots"]) - 1
    else:
        legend_index = None

    fig, axes = plt.subplots(*grid_shape, figsize=figsize)

    # Determine if we should include xlabels or ylabels
    use_xlabels = any(["xlabel" in plot.get("kwargs", {}) for plot in config["plots"].values()])
    use_ylabels = any(["ylabel" in plot.get("kwargs", {}) for plot in config["plots"].values()])

    for i, (plot_title, plot_config) in enumerate(config["plots"].items()):
        y_index, x_index = i // grid_shape[1], i % grid_shape[1]
        ax = axes.flat[i] if len(config["plots"]) > 1 else axes

        paths, labels = list(plot_config["methods"].values()), list(plot_config["methods"].keys())
        plot_title = plot_title if config.get("use_subplot_titles") else None
        plot_kwargs = config.get("kwargs", dict()).copy()
        plot_kwargs.update(plot_config.get("kwargs", {}))

        create_plot(paths, labels, ax, plot_title, color_map=config.get("color_map"), **plot_kwargs)

        if x_index != 0 and not use_ylabels:
            ax.set_ylabel(None)
        if y_index != grid_shape[0] - 1 and not use_xlabels:
            ax.set_xlabel(None)
        if i != legend_index and legend_pos != "all":
            ax.get_legend().remove()
        else:
            ax.legend(frameon=False, loc="lower right")

        # Check to see if we can place an image in the corner of the plot.
        if plot_config.get("image") is not None:
            import matplotlib.image as mpimg

            # use inset axes to create an inset image
            axins = inset_axes(ax, width="35%", height="35%", loc=4, borderpad=0.2)
            image = mpimg.imread(plot_config["image"])
            axins.imshow(image)
            axins.axis("off")

    if config.get("title"):
        fig.suptitle(config.get("title"), y=1.0)
        rect[3] -= 0.01

    # If the legend is set to the bottom do it here
    if legend_pos == "bottom":
        bbox_offset = -0.07
        rect_offset = 0.11
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower left",
            ncol=len(handles),
            bbox_to_anchor=(0.0, bbox_offset / figsize[1]),
            frameon=False,
        )
        rect[1] += rect_offset / figsize[1]

    plt.tight_layout(pad=0, rect=rect)
