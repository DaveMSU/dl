import argparse
import json
import logging
import pathlib
import sys

from numpy import seterr

from .learning_config import LearningConfig
from .trainer import Trainer


def train_add_cmdargs(
        parser: argparse.ArgumentParser,
        subparsers: argparse._SubParsersAction
) -> None:
    p = subparsers.add_parser(
        "train",
        help=(
            "Train a model on specified data."
        )
    )
    p.set_defaults(main=train_main)

    p.add_argument(
        "-n", "--net_factory_function_impl",
        required=True,
        type=pathlib.Path,
        help=(
            "POSIX path to the *.py file, that has only one object -"
            " function named 'create_a_neural_network_instance', that"
            " returns torch.nn.Module subclass instance - the neural network"
        )
    )

    p.add_argument(
        "-l", "--learning_config",
        required=True,
        type=pathlib.Path,
        help=(
            "POSIX path to the *.json config file that fully describes the"
            " learning process. For examples see configs"
            " ./exps/*/learning_config.json"
        )
    )

    p.add_argument(
        "--log-level",
        default="INFO",
        type=str,
        help="Logging level, possible options: 'INFO', 'DEBUG', 'TRACE'"
    )


def train_main(cmd_args: argparse.Namespace) -> None:
    logging.basicConfig(
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
        level=getattr(logging, cmd_args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    if cmd_args.log_level == "INFO":
        seterr(divide='ignore', invalid='ignore')

    with open(cmd_args.learning_config, "r") as lf:
        trainer = Trainer(
            net_factory_function_path=cmd_args.net_factory_function_impl,
            learning_config=LearningConfig.from_dict(json.load(lf))
        )
    trainer.run()
