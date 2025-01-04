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
        help=(
            "The command is used to prepare an hd5f dataset before allowing"
            " the dataloader to sample batches from it and apply it's own"
            " transforms while training, the source hd5f dataset must"
            " contain only two columns: 'input' and 'output'"
        )
    )
    p.set_defaults(main=wrangle_dataset_main)

    p.add_argument(
        "-c", "--config",
        required=True,
        type=str,
        help=(
            "POSIX path to the *.json config file that specifies the way of"
            " how to wrangle the datasets, the config mush be as follows:"
            "{"
            "    'src_dataset_path': str,"
            "    'dst_dataset_path': str,"
            "    'wrangler': {"
            "        'type': str,"
            "        'params': tp.Dict[str, tp.Any]"
            "    }"
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
