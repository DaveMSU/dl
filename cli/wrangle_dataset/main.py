import argparse
import json

from .impl.wrangler_factory import wrangler_factory
from .impl.wranglers import BaseWrangler
from .impl.wrangling_config import WranglingConfig


def wrangle_dataset_add_cmdargs(
        parser: argparse.ArgumentParser,
        subparsers: argparse._SubParsersAction
) -> None:
    p = subparsers.add_parser(
        "wrangle_dataset",
        help="Takes raw h5 dataset and make it more prepared for dataloader"
    )
    p.set_defaults(main=wrangle_dataset_main)

    p.add_argument(
        "-c", "--config",
        required=True,
        type=str,
        help=(
            "POSIX path to the json file that specifies a behaviour of"
            " wrangling of the datasets, the config is expected to be like:"
            "{"  # TODO: write it
            "}"
        )
    )


def wrangle_dataset_main(cmd_args: argparse.Namespace) -> None:
    with open(cmd_args.config, "r") as f:
        wrangling_config = WranglingConfig.from_dict(json.load(f))
        wrangler: BaseWrangler = wrangler_factory(
            wrangling_config.src_dataset_path,
            wrangling_config.dst_dataset_path,
            wrangling_config.wrangler_config
        )
        wrangler.run()
