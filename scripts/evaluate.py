import argparse
import os

from research.utils.config import Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model")
    parser.add_argument("--path", type=str, default=None, required=False, help="Path to save the gif")
    parser.add_argument("--device", "-d", type=str, default="auto")
    parser.add_argument("--num-ep", type=int, default=1, help="Number of episodes")
    parser.add_argument("--num-gifs", type=int, default=0, help="Number of gifs to save.")
    parser.add_argument("--every-n-frames", type=int, default=2, help="Save every n frames to the gif.")
    parser.add_argument("--width", type=int, default=160, help="Width of image")
    parser.add_argument("--height", type=int, default=120, help="Height of image")
    parser.add_argument("--strict", action="store_true", default=False, help="Strict")
    parser.add_argument(
        "--terminate-on-success", action="store_true", default=False, help="Terminate gif on success condition."
    )
    parser.add_argument(
        "--override",
        metavar="KEY=VALUE",
        nargs="+",
        default=[],
        help="Set kv pairs used as args for the entry point script.",
    )
    parser.add_argument("--max-len", type=int, default=1000, help="maximum length of an episode.")
    args = parser.parse_args()

    assert args.checkpoint.endswith(".pt"), "Must provide a model checkpoint"
    config = Config.load(os.path.dirname(args.checkpoint))
    config["checkpoint"] = None  # Set checkpoint to None, we don't actually need to load it.

    if args.path is None:
        args.path = os.path.dirname(args.checkpoint)

    # Overrides
    print("Overrides:")
    for override in args.override:
        print(override)

    # Overrides
    for override in args.override:
        items = override.split("=")
        key, value = items[0].strip(), "=".join(items[1:])
        # Progress down the config path (seperated by '.') until we reach the final value to override.
        config_path = key.split(".")
        config_dict = config
        while len(config_path) > 1:
            config_dict = config_dict[config_path[0]]
            config_path.pop(0)
        config_dict[config_path[0]] = value

    if len(args.override) > 0:
        print(config)

    # Make sure we don't use subprocess evaluation
    config["trainer_kwargs"]["eval_env_runner"] = None

    # Over-write the parameters in the eval_kwargs
    assert config["trainer_kwargs"]["eval_fn"] == "eval_policy", "Evaluate only works with eval_policy for now."
    config["trainer_kwargs"]["eval_kwargs"]["num_ep"] = args.num_ep
    config["trainer_kwargs"]["eval_kwargs"]["num_gifs"] = args.num_gifs
    config["trainer_kwargs"]["eval_kwargs"]["width"] = args.width
    config["trainer_kwargs"]["eval_kwargs"]["height"] = args.height
    config["trainer_kwargs"]["eval_kwargs"]["every_n_frames"] = args.every_n_frames
    config["trainer_kwargs"]["eval_kwargs"]["terminate_on_success"] = args.terminate_on_success
    config = config.parse()
    model = config.get_model(device=args.device)
    metadata = model.load(args.checkpoint)
    trainer = config.get_trainer(model=model)
    # Run the evaluation loop
    os.makedirs(args.path, exist_ok=True)
    metrics = trainer.evaluate(args.path, metadata["step"])

    print("[research] Eval policy finished:")
    for k, v in metrics.items():
        print(k, v)
