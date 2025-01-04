import dataclasses
import pathlib
import typing as tp


@dataclasses.dataclass(frozen=True)
class WranglerConfig:
    type: str
    params: tp.Dict[str, tp.Any]  # kwargs


@dataclasses.dataclass(frozen=True)
class WranglingConfig:
    src_dataset_path: pathlib.PosixPath
    dst_dataset_path: pathlib.PosixPath
    wrangler_config: WranglerConfig

    @classmethod
    def from_dict(cls, d: tp.Dict[str, tp.Any]) -> 'WranglingConfig':
        return cls(
            src_dataset_path=pathlib.Path(d["src_dataset_path"]),
            dst_dataset_path=pathlib.Path(d["dst_dataset_path"]),
            wrangler_config=WranglerConfig(
                type=d["wrangler"]["type"],
                params=d["wrangler"]["params"]
            )
        )
