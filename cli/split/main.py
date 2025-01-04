import argparse
import pathlib

from .impl.split_mode import SplitMode
from .impl.splitters.factory import splitter_factory
from lib.types import Ratio


def split_add_cmdargs(
        parser: argparse.ArgumentParser,
        subparsers: argparse._SubParsersAction
) -> None:
    p = subparsers.add_parser(
        "split",
        help=(
            "The command is used to split an hd5f dataset into two; the"
            " dataset must contain only two columns: 'input' and 'output'"
            " (this tool isn't the best way to deal with it, 'cause the best"
            " way to produce all of its results is to run SQL queries under"
            " tables, not the code under *.h5 files)"
        )
    )
    p.set_defaults(main=split_main)

    p.add_argument(
        "--src",
        required=True,
        type=pathlib.Path,
        help="Path to the source hd5f dataset on disk"
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
        "--th",
        required=True,
        type=Ratio,
        help=(
            "The threshold that specifies which ratio of the source dataset"
            " will be in the first part of splitting (th) and which one"
            " will be in the second one (1 - th)"
        )
    )

    p.add_argument(
        "--mode",
        required=True,
        type=SplitMode,
        help=(
            "The split mode. F.e., 'dummy' is without mixing anything, just"
            " take 'th' of the dataset from above; 'random' is the same, but"
            " after mixing the indexes before that;"
            " see the others in the code."
        )
    )


def split_main(cmd_args: argparse.Namespace) -> None:
    Splitter: type = splitter_factory(cmd_args.mode)
    Splitter(
        src=cmd_args.src,
        dst0=cmd_args.dst0,
        dst1=cmd_args.dst1,
        threshold=cmd_args.th,
    ).run()
