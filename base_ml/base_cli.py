# -*- coding: utf-8 -*-
# Base CLI to parse Arguments
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import argparse
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Union

import yaml
from pydantic import BaseModel


class ABCParser(ABC):
    """Blueprint for Argument Parser"""

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_config(self) -> Tuple[Union[BaseModel, dict], logging.Logger]:
        """Load configuration and create a logger

        Returns:
            Tuple[PreProcessingConfig, logging.Logger]: Configuration and Logger
        """
        pass

    @abstractmethod
    def store_config(self) -> None:
        """Store the config file in the logging directory to keep track of the configuration."""
        pass


class ExperimentBaseParser:
    """Configuration Parser for Machine Learning Experiments"""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Start an experiment with given configuration file.",
        )
        requiredNamed = parser.add_argument_group("required named arguments")
        requiredNamed.add_argument(
            "--config", type=str, help="Path to a config file", required=True
        )
        parser.add_argument("--gpu", type=int, help="Cuda-GPU ID")
        parser.add_argument(
            "--checkpoint",
            type=str,
            help="Path to a PyTorch checkpoint file. "
            "The file is loaded and continued to train with the provided settings. "
            "This parameter cannot be set in the config file!",
        )

        self.parser = parser

    def parse_arguments(self) -> Tuple[Union[BaseModel, dict]]:
        """Parse the arguments from CLI and load yaml config

        Returns:
            Tuple[Union[BaseModel, dict]]: Parsed arguments
        """
        # parse the arguments
        opt = self.parser.parse_args()
        with open(opt.config, "r") as config_file:
            yaml_config = yaml.safe_load(config_file)
            yaml_config_dict = dict(yaml_config)

        opt_dict = vars(opt)
        # the cli gpu argument overwrites the config file
        if opt_dict.get("gpu") is not None:
            yaml_config_dict["gpu"] = opt_dict["gpu"]

        # a checkpoint can only be passed via cli; legacy wandb sweep keys are ignored
        for legacy_key in ("run_sweep", "agent", "checkpoint"):
            yaml_config_dict.pop(legacy_key, None)
        if opt_dict.get("checkpoint") is not None:
            yaml_config_dict["checkpoint"] = opt_dict["checkpoint"]

        self.config = yaml_config_dict

        return self.config
