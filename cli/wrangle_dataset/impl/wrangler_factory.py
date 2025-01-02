import dataclasses
import pathlib

from . import wranglers
from .wrangling_config import WranglerConfig
from lib.sample_transforms.factory import transform_factory
from lib.types import (
    TransformConfig,
)


def wrangler_factory(
        src_dataset_path: pathlib.PosixPath,
        dst_dataset_path: pathlib.PosixPath,
        wrangler_config: WranglerConfig
) -> wranglers.BaseWrangler:
    if wrangler_config.type == "TransformsApplier":
        return getattr(wranglers, "TransformsApplier")(
            src_dataset_path=src_dataset_path,
            dst_dataset_path=dst_dataset_path,
            transforms=tuple(
                transform_factory(TransformConfig.from_dict(d))
                for d in wrangler_config.params["transforms"]
            )
        )
    else:
        raise TypeError(f"No such class as `{WranglerConfig.type}`")
