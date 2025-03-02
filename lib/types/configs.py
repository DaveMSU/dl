import dataclasses
import typing as tp


@dataclasses.dataclass(frozen=True)
class _BaseConfig:
    type: str
    params: tp.Dict[str, tp.Any]  # kwargs

    @classmethod
    def from_dict(cls, d: tp.Dict[str, tp.Any]) -> '_BaseConfig':
        return cls(
            type=d["type"],
            params=d["params"]
        )


@dataclasses.dataclass(frozen=True)
class TransformConfig(_BaseConfig):
    pass


@dataclasses.dataclass(frozen=True)
class TokenizerConfig(_BaseConfig):
    pass
