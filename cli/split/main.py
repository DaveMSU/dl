import argparse
import json
import pathlib

from .impl.split_mode import SplitMode
from .impl.splitter import Splitter
from lib.types import Ratio


def split_add_cmdargs(
        parser: argparse.ArgumentParser,
        subparsers: argparse._SubParsersAction
) -> None:
    p = subparsers.add_parser(
        "split",
        help=(
            "The command uses to split an h5py dataset to two;"
            " the dataset must contain only two columns: input and output"
            " (this tool is tmp, 'cause the best way to produce all of its"
            " results is to run SQL queries under tables, not h5py files)"
        )
    )
    p.set_defaults(main=split_main)

    p.add_argument(
        "--src",
        required=True,
        type=pathlib.Path,
        help="Path to the source h5py dataset on the disk"
    )

    p.add_argument(
        "--dst0",
        required=True,
        type=pathlib.Path,
        help=(
            "Destination path on the disk to the first"
            " splited part of the source dataset"
        )
    )

    p.add_argument(
        "--dst1",
        required=True,
        type=pathlib.Path,
        help=(
            "Destination path on the disk to the second"
            " splited part of the source dataset"
        )
    )

    p.add_argument(
        "-th", "--threshold",
        required=True,
        type=Ratio,
        help=(
            "The threshold that specifies which ratio of the source dataset"
            " will be in the first part of splitting (th) and which one"
            " will be in the second one (1 - th)"
        )
    )

    p.add_argument(
        "-m", "--mode",
        required=True,
        type=SplitMode,
        help="Choose the enum type of the dataset splitting"
    )
    

def split_main(cmd_args: argparse.Namespace) -> None:
    Splitter(
        src=cmd_args.src,
        dst0=cmd_args.dst0,
        dst1=cmd_args.dst1,
        threshold=cmd_args.threshold,
        split_mode=cmd_args.mode
    ).run()
