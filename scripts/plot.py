import argparse
import os

from matplotlib import pyplot as plt

from research.utils import plotter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", type=str, default="plot.png", help="Path of output plot")
    parser.add_argument("--path", "-p", nargs="+", type=str, required=True, help="Paths of runs to plot")
    parser.add_argument(
        "--legend", "-l", nargs="+", type=str, required=False, help="Names of each run to display in the legend"
    )
    parser.add_argument("--title", "-t", type=str, required=False, help="Plot title")
    parser.add_argument("--window", "-w", type=int, default=1, help="Moving window averaging parameter.")
    parser.add_argument("--x", "-x", type=str, default="step", help="X value to plot")
    parser.add_argument("--max-x", "-m", type=int, default=None, help="Max x value to plot")
    parser.add_argument("--xlabel", "-xl", type=str, default=None, help="X label to display on the plot")
    parser.add_argument("--y", "-y", type=str, nargs="+", default=["eval/loss"], help="Y value(s) to plot")
    parser.add_argument("--ylabel", "-yl", type=str, default=None, help="Y label to display on the plot")
    parser.add_argument("--fig-size", "-f", nargs=2, type=int, default=(3, 2))
    args = parser.parse_args()

    paths = args.path

    if len(paths) == 1 and paths[0].endswith(".yaml"):
        # We are creating a plot via config
        plotter.plot_from_config(paths[0])
        plt.savefig(args.output, dpi=300)  # Increase DPI for higher res.
    else:
        # Check to see if we should auto-expand the path.
        # Do this only if the number of paths specified is one and each sub-path is a directory
        if len(paths) == 1 and all([os.path.isdir(os.path.join(paths[0], d)) for d in os.listdir(paths[0])]):
            paths = sorted([os.path.join(paths[0], d) for d in os.listdir(paths[0])])
        # Now create the labels
        labels = args.legend
        if labels is None:
            labels = [os.path.basename(path[:-1] if path.endswith("/") else path) for path in paths]
        # Sort the paths alphabetically by the labels
        paths, labels = zip(*sorted(zip(paths, labels), key=lambda x: x[0]))  # Alphabetically sort by filename

        plotter.create_plot(
            paths,
            labels,
            title=args.title,
            xlabel=args.xlabel,
            ylabel=args.ylabel,
            x_key=args.x,
            y_keys=args.y,
            window_size=args.window,
            max_x_value=args.max_x,
        )

        # Save the plot
        print("[research] Saving plot to", args.output)
        plt.gcf().set_size_inches(*args.fig_size)
        plt.tight_layout(pad=0)
        plt.savefig(args.output, dpi=300)  # Increase DPI for higher res.
