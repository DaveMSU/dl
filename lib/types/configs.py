import dataclasses
import typing as tp


@dataclasses.dataclass(frozen=True)
class TransformConfig:
    type: str
    params: tp.Dict[str, tp.Any]  # kwargs

    @classmethod
    def from_dict(cls, d: tp.Dict[str, tp.Any]) -> 'TransformConfig':
        return cls(
            type=d["type"],
            params=d["params"]
        )
