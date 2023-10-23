import argparse
import collections
import os

import numpy as np
import pandas as pd

from research.utils.plotter import LOG_FILE_NAME, moving_avg


def get_score(path, y_key, window=1, use_max=False):
    # Get all of the seeds
    if LOG_FILE_NAME in os.listdir(path):
        paths = [path]
    else:
        assert all(["seed-" in p for p in os.listdir(path)])
        paths = [os.path.join(path, seed) for seed in os.listdir(path)]
    values = collections.defaultdict(list)
    for p in paths:
        df = pd.read_csv(os.path.join(p, LOG_FILE_NAME))
        if y_key not in df:
            print("[tools] Error: key", y_key, "not in", p)
        x, y = moving_avg(df["step"].to_numpy(), df[y_key].to_numpy(), window)
        for i in range(len(x)):
            values[x[i]].append(y[i])
    # Compute the final values by averaging
    values = {k: np.mean(v) for k, v in values.items()}
    if use_max:
        return max(values.values())
    else:
        return min(values.values())


def get_params(path):
    # Separate out the hyperparamters
    name = os.path.basename(path)
    parts = name.split("_")
    params = {}
    for part in parts:
        split_part = part.split("-")
        name, value = split_part[0], "-".join(split_part[1:])
        params[name] = value
    return params


def get_paths(path):
    files = os.listdir(path)
    if LOG_FILE_NAME in files or any(["seed" in f for f in files]):
        return [path]
    else:
        return sum([get_paths(os.path.join(path, f)) for f in files], start=[])


if __name__ == "__main__":
    # Do something
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to the run to handle.")
    parser.add_argument("--y-key", type=str, default="step", help="name of string to parse")
    parser.add_argument("--window", type=int, default=1, help="averaging window")
    parser.add_argument("--use-max", action="store_true", help="if we should return the max for each metric")
    parser.add_argument("--include", nargs="+", type=str, default=[], help="Include filters")
    parser.add_argument("--exclude", nargs="+", type=str, default=[], help="Exclude filters")
    args = parser.parse_args()

    paths = get_paths(args.path)
    # Run the path filters
    for include in args.include:
        paths = [path for path in paths if include in path]
    for exclude in args.exclude:
        paths = [path for path in paths if exclude not in path]
    params_list = [get_params(path) for path in paths]
    scores = [get_score(path, args.y_key, window=args.window, use_max=args.use_max) for path in paths]

    # Get all hyperparameter configurations
    hyperparameters = collections.defaultdict(set)
    for params in params_list:
        for name, value in params.items():
            hyperparameters[name].add(value)

    # For each hyperparameter, construct a report for its values averaged over scores
    for param, values in hyperparameters.items():
        print("[Sweep Parser] Ablating parameter", param)
        for value in sorted(values):
            avg_score = np.mean(
                [scores[i] for i in range(len(scores)) if param in params_list[i] and params_list[i][param] == value]
            )
            print(value, ":", avg_score)
