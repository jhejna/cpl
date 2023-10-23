# Contrastive Preference Learning: Learning from Human Feedback without RL

This is the official codebase for [*Contrastive Preference Learning: Learning From Human Feedback without RL*](https://arxiv.org/abs/2310.13639) by Joey Hejna, Rafael Rafailov\*, Harshit Sikchi\*, Chelsea Finn, Scott Niekum, W. Bradley Knox, and Dorsa Sadigh.

Below we include instructions for reproducing results found in the paper. This repository is based on a frozen version of [research-lightning](https://github.com/jhejna/research-lightning). For detailed information about how to use the repository, refer to that repository.

If you find our paper or code insightful, feel free to cite us with the following bibtex:
```
@InProceedings{hejna23contrastive,
  title = {Contrastive Preference Learning: Learning From Human Feedback without RL},
  author = {Hejna, Joey and Rafailov, Rafael and Sikchi, Harshit and Finn, Chelsea and Niekum, Scott and Knox, W. Bradley and Sadigh, Dorsa},
  booktitle = {ArXiv preprint},
  year = {2023},
  url = {https://arxiv.org/abs/2310.13639}
}
```

## Installation

Complete the following steps:
1. Clone the repository to your desired location using `git clone`.
2. Create the conda environment using `conda env create -f environment_<cpu or gpu>.yaml`. Note that the correct MetaWorld version must be used.
3. Install the repository research package via `pip install -e research`.
4. Modify the `setup_shell.sh` script by updating the appropriate values as needed. The `setup_shell.sh` script should load the environment, move the shell to the repository directory, and additionally setup any external dependencies. All the required flags should be at the top of the file. This is necesary for support with the SLURM launcher, which we used to run experiments.
5. Download the metaworld datasets [here](https://drive.google.com/file/d/1lo5Wt9Go_E_5c8ymfXFvsTY6kDUelqu2/view?usp=share_link). Extract the files into a `datasets` folder in the repository root. This should match the paths in the config files.

When using the repository, you should be able to setup the environment by running `. path/to/setup_shell.sh`.

## Usage

To train a model, simply run `python scripts/train.py --config path/to/config --path path/to/save/folder` after activating the environment.

Multiple experiments can be run at a single time using a `.json` sweep file. To run a sweep, first create one, then run a sweep command using either `tools/run_slurm.py` or `tools/run_local.py`. Specify the slurm config and output directory with `--arguments config=path/to/config path=path/to/save/folder`. For example sweep files, checkout the Inverse Preference Learning [repository](https://github.com/jhejna/inverse-preference-learning).


## License

This has an MIT license, found in the [LICENSE](LICENSE) file.
