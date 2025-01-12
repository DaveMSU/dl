import argparse
import json
import pathlib

from .impl.tokenizer_factory import tokenizer_factory
from .impl.tokenizer_training_config import TokenizerTrainingConfig
from lib.types.tokenizers import BaseTokenizer


def train_tokenizer_add_cmdargs(
        parser: argparse.ArgumentParser,
        subparsers: argparse._SubParsersAction
) -> None:
    p = subparsers.add_parser(
        "train_tokenizer",
        help=(
            "Train a tokenizer to produce tokens (tp.Sequence[int])"
            " for a given text (str)"
        )
    )
    p.set_defaults(main=train_tokenizer_main)

    p.add_argument(
        "-c", "--config",
        required=True,
        type=pathlib.Path,
        help=(
            "POSIX path to the *.json config file that specifies the"
            " tokenizer type, its parameters and how to train it. For"
            " examples see configs ./exps/*/tokenization_config.json"
        )
    )


def train_tokenizer_main(cmd_args: argparse.Namespace) -> None:
    with open(cmd_args.config, "r") as f:
        config = TokenizerTrainingConfig.from_dict(json.load(f))
        tokenizer: BaseTokenizer = tokenizer_factory(config.tokenizer_config)
        tokenizer.fit(config.train_dataset.path, config.train_dataset.limit)
        tokenizer.save(config.save_path)
