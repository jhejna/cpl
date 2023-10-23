import copy
import functools
import gc
import importlib
import os
import pprint
import random
from typing import Any, Dict, Optional, Union

import gym
import numpy as np
import torch
import yaml

import research

from . import schedules, utils
from .trainer import Trainer

DEFAULT_NETWORK_KEY = "network"


def get_env(env: str, env_kwargs: Dict, wrapper: str, wrapper_kwargs: Dict) -> gym.Env:
    # Try to get the environment
    try:
        env = vars(research.envs)[env](**env_kwargs)
    except KeyError:
        env = gym.make(env, **env_kwargs)
    if wrapper is not None:
        env = vars(research.envs)[wrapper](env, **wrapper_kwargs)
    return env


class BareConfig(object):
    """
    This is a bare copy of the config that does not require importing any of the research packages.
    This file has been copied out for use in the tools/trainer etc. to avoid loading heavy packages
    when the goal is just to create configs. It defines no structure.

    There is one caviat: the Config is designed to handle import via a custom ``import'' key.
    This is handled ONLY at load time.
    """

    def __init__(self):
        # Define the necesary structure for a complete training configuration
        self._parsed = False
        self.config = dict()

    def save(self, path: str) -> None:
        if os.path.isdir(path):
            path = os.path.join(path, "config.yaml")
        with open(path, "w") as f:
            yaml.dump(self.config, f)

    def update(self, d: Dict) -> None:
        self.config.update(d)

    @classmethod
    def load(cls, path: str) -> "BareConfig":
        if os.path.isdir(path):
            path = os.path.join(path, "config.yaml")
        with open(path, "r") as f:
            data = yaml.load(f, Loader=yaml.Loader)
        # Check for imports
        config = cls()
        if "import" in data:
            imports = data["import"]
            imports = [imports] if not isinstance(imports, list) else imports
            # Load the imports in order
            for import_path in imports:
                config.update(BareConfig.load(import_path).config)
            del data["import"]
        config.update(data)
        assert "import" not in config
        return config

    def get(self, key: str, default: Optional[Any] = None):
        return self.config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.config[key] = value

    def __contains__(self, key: str):
        return self.config.__contains__(key)

    def __str__(self) -> str:
        return pprint.pformat(self.config, indent=4)

    def copy(self) -> "Config":
        assert not self.parsed, "Cannot copy a parsed config"
        config = type(self)()
        config.config = copy.deepcopy(self.config)
        return config


class Config(BareConfig):
    def __init__(self):
        super().__init__()
        # Define necesary fields

        # Manage seeding.
        self._seeded = False
        self.config["seed"] = None

        # Env Args
        self.config["env"] = None
        self.config["env_kwargs"] = {}

        self.config["eval_env"] = None
        self.config["eval_env_kwargs"] = {}

        self.config["wrapper"] = None
        self.config["wrapper_kwargs"] = {}

        # Algorithm Args
        self.config["alg"] = None
        self.config["alg_kwargs"] = {}

        # Dataset Args
        self.config["dataset"] = None
        self.config["dataset_kwargs"] = {}

        self.config["validation_dataset"] = None
        self.config["validation_dataset_kwargs"] = None

        # Processor arguments
        self.config["processor"] = None
        self.config["processor_kwargs"] = {}

        # Optimizer Args
        self.config["optim"] = None
        self.config["optim_kwargs"] = {}

        # Network Args
        self.config["network"] = None
        self.config["network_kwargs"] = {}

        # Checkpoint
        self.config["checkpoint"] = None

        # Schedule args
        self.config["schedule"] = None
        self.config["schedule_kwargs"] = {}

        self.config["trainer_kwargs"] = {}

    @property
    def parsed(self):
        return self._parsed

    @staticmethod
    def _parse_helper(d: Dict) -> None:
        for k, v in d.items():
            if isinstance(v, list) and len(v) > 1 and v[0] == "import":
                # parse the value to an import
                d[k] = getattr(importlib.import_module(v[1]), v[2])
            elif isinstance(v, dict):
                Config._parse_helper(v)

    def parse(self) -> "Config":
        config = self.copy()
        Config._parse_helper(config.config)
        config._parsed = True
        # Before we make any objects, make sure we set the seeds.
        if self.config["seed"] is not None:
            torch.manual_seed(self.config["seed"])
            np.random.seed(self.config["seed"])
            random.seed(self.config["seed"])
        return config

    def flatten(self, separator=".") -> Dict:
        """Returns a flattened version of the config where '.' separates nested values"""
        return utils.flatten_dict(self.config, separator=separator)

    def __setitem__(self, key: str, value: Any):
        if key not in self.config:
            raise ValueError(
                "Attempting to set an out of structure key: " + key + ". Configs must follow the format in config.py"
            )
        super().__setitem__(key, value)

    def get_train_env_fn(self):
        """
        Returns a function that generates a training environment, or None if no training environment is used.
        """
        assert self.parsed
        if self["env"] is None:
            return None
        else:
            return functools.partial(
                get_env,
                env=self["env"],
                env_kwargs=self["env_kwargs"],
                wrapper=self["wrapper"],
                wrapper_kwargs=self["wrapper_kwargs"],
            )

    def get_eval_env_fn(self):
        """
        Returns a function that generates an evaluation environment.
        Will always return an environment.
        """
        assert self.parsed
        # Return the evalutaion environment.
        if self["eval_env"] is None:
            env, env_kwargs = self["env"], self["env_kwargs"]
        else:
            env, env_kwargs = self["eval_env"], self["eval_env_kwargs"]
        return functools.partial(
            get_env, env=env, env_kwargs=env_kwargs, wrapper=self["wrapper"], wrapper_kwargs=self["wrapper_kwargs"]
        )

    def get_spaces(self):
        # Try to get the spaces. Eval env will always return a space.
        dummy_env = self.get_eval_env_fn()()  # Call the function.
        observation_space = utils.space_copy(dummy_env.observation_space)
        action_space = utils.space_copy(dummy_env.action_space)
        dummy_env.close()
        del dummy_env
        gc.collect()
        return observation_space, action_space

    def get_model(
        self,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        device: Union[str, torch.device] = "auto",
    ):
        assert self.parsed

        if observation_space is None or action_space is None:
            observation_space, action_space = self.get_spaces()

        # This function returns the model
        alg_class = vars(research.algs)[self["alg"]]
        dataset_class = None if self["dataset"] is None else vars(research.datasets)[self["dataset"]]
        validation_dataset_class = (
            None if self["validation_dataset"] is None else vars(research.datasets)[self["validation_dataset"]]
        )
        network_class = None if self["network"] is None else vars(research.networks)[self["network"]]
        optim_class = None if self["optim"] is None else vars(torch.optim)[self["optim"]]
        processor_class = None if self["processor"] is None else vars(research.processors)[self["processor"]]

        # Fetch the schedulers. If we don't have an schedulers dict, change it to one.
        if not isinstance(self["schedule"], dict):
            schedulers_class = {DEFAULT_NETWORK_KEY: self["schedule"]}
            schedulers_kwargs = {DEFAULT_NETWORK_KEY: self["schedule_kwargs"]}
        else:
            schedulers_class = self["schedule"]
            schedulers_kwargs = self["schedule_kwargs"]

        # Make sure we fetch the schedule if its provided as a string
        for k in schedulers_class.keys():
            if isinstance(schedulers_class[k], str):
                # Create the lambda function, and pass it in as a keyword arg
                schedulers_kwargs[k] = dict(lr_lambda=vars(schedules)[schedulers_class[k]](**schedulers_kwargs[k]))
                schedulers_class[k] = torch.optim.lr_scheduler.LambdaLR

        algo = alg_class(
            observation_space,
            action_space,
            network_class,
            dataset_class,
            network_kwargs=self["network_kwargs"],
            dataset_kwargs=self["dataset_kwargs"],
            validation_dataset_class=validation_dataset_class,
            validation_dataset_kwargs=self["validation_dataset_kwargs"],
            processor_class=processor_class,
            processor_kwargs=self["processor_kwargs"],
            optim_class=optim_class,
            optim_kwargs=self["optim_kwargs"],
            schedulers_class=schedulers_class,
            schedulers_kwargs=schedulers_kwargs,
            checkpoint=self["checkpoint"],
            device=device,
            **self["alg_kwargs"],
        )
        return algo

    def get_trainer(
        self,
        model=None,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        device: Union[str, torch.device] = "auto",
    ):
        if model is None:
            if observation_space is None or action_space is None:
                observation_space, action_space = self.get_spaces()
            model = self.get_model(observation_space=observation_space, action_space=action_space, device=device)
        train_env_fn = self.get_train_env_fn()
        eval_env_fn = self.get_eval_env_fn()
        # Return the trainer...
        return Trainer(model, train_env_fn, eval_env_fn, **self["trainer_kwargs"])
