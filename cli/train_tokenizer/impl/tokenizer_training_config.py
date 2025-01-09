import dataclasses
import pathlib
import typing as tp

from lib.types import TokenizerConfig


@dataclasses.dataclass(frozen=True)
class TokenizerTrainingConfig:
    save_path: pathlib.PosixPath
    train_dataset_path: pathlib.PosixPath
    tokenizer_config: TokenizerConfig

    @classmethod
    def from_dict(cls, d: tp.Dict[str, tp.Any]) -> 'TokenizerTrainingConfig':
        return cls(
            save_path=pathlib.Path(d["save_path"]),
            train_dataset_path=pathlib.Path(d["train_dataset_path"]),
            tokenizer_config=TokenizerConfig(**d["tokenizer_config"])
        )
