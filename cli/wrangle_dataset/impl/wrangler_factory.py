import pathlib

from . import wranglers
from .wrangling_config import WranglerConfig
from lib.sample_transforms.factory import transform_factory
from lib.types import (
    TokenizerConfig,
    TransformConfig,
)
from lib.types import tokenizers


def wrangler_factory(
        src_dataset_path: pathlib.PosixPath,
        dst_dataset_path: pathlib.PosixPath,
        wrangler_config: WranglerConfig
) -> wranglers.BaseWrangler:
    if wrangler_config.type == "TokenizerApplier":
        _cnfg = TokenizerConfig.from_dict(wrangler_config.params["tokenizer"])
        _tokenizer = getattr(tokenizers, _cnfg.type)(**_cnfg.params)
        if wrangler_config.params["load_path"] is not None:
            _tokenizer.load(
                pathlib.Path(wrangler_config.params["load_path"])
            )
        return getattr(wranglers, "TokenizerApplier")(
            src_dataset_path=src_dataset_path,
            dst_dataset_path=dst_dataset_path,
            tokenizer=_tokenizer
        )
    elif wrangler_config.type == "TransformsApplier":
        return getattr(wranglers, "TransformsApplier")(
            src_dataset_path=src_dataset_path,
            dst_dataset_path=dst_dataset_path,
            transforms=tuple(
                transform_factory(TransformConfig.from_dict(d))
                for d in wrangler_config.params["transforms"]
            )
        )
    else:
        raise TypeError(f"No such class as `{wrangler_config.type}`")
