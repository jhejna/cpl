import argparse
import os
import subprocess

from research.utils.config import Config


def try_wandb_setup(path, config):
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key is not None and wandb_api_key != "":
        try:
            import wandb
        except ImportError:
            return
        project_dir = os.path.dirname(os.path.dirname(__file__))
        wandb.init(
            project=os.path.basename(project_dir),
            name=os.path.basename(path),
            config=config.flatten(separator="-"),
            dir=os.path.join(os.path.dirname(project_dir), "wandb"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default=None)
    parser.add_argument("--path", "-p", type=str, default=None)
    parser.add_argument("--device", "-d", type=str, default="auto")
    args = parser.parse_args()

    config = Config.load(args.config)
    print(config)
    os.makedirs(args.path, exist_ok=False)  # Change this to false temporarily so we don't recreate experiments
    try_wandb_setup(args.path, config)
    config.save(args.path)  # Save the config
    # save the git hash
    process = subprocess.Popen(["git", "rev-parse", "HEAD"], shell=False, stdout=subprocess.PIPE)
    git_head_hash = process.communicate()[0].strip()
    with open(os.path.join(args.path, "git_hash.txt"), "wb") as f:
        f.write(git_head_hash)
    # Parse the config file to resolve names.
    config = config.parse()
    # Get everything at once.
    trainer = config.get_trainer(device=args.device)
    # Train the model
    trainer.train(args.path)
