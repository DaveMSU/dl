import dataclasses
import pathlib
import typing as tp

from lib.types import TokenizerConfig


@dataclasses.dataclass(frozen=True)
class _TrainingDatasetConfig:
    path: pathlib.PosixPath
    limit: int


@dataclasses.dataclass(frozen=True)
class TokenizerTrainingConfig:
    train_dataset: _TrainingDatasetConfig
    tokenizer_config: TokenizerConfig
    save_path: pathlib.PosixPath

    @classmethod
    def from_dict(cls, d: tp.Dict[str, tp.Any]) -> 'TokenizerTrainingConfig':
        return cls(
            train_dataset=_TrainingDatasetConfig(
                path=d["train_dataset"]["path"],
                limit=d["train_dataset"]["limit"]
            ),
            tokenizer_config=TokenizerConfig(**d["tokenizer"]),
            save_path=pathlib.Path(d["save_path"])
        )
