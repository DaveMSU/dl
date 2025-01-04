import dataclasses
import enum
import pathlib
import typing as tp

from lib.types import TransformConfig


@dataclasses.dataclass(frozen=True)
class _OneDataloaderConfig:  # TODO: add accumulation
    dataset_path: pathlib.PosixPath
    batch_size: int
    shuffle: bool
    drop_last: bool
    transforms: tp.List[TransformConfig]


@dataclasses.dataclass(frozen=True)
class _TrainValDataloadersConfig:
    train: _OneDataloaderConfig
    val: _OneDataloaderConfig


@dataclasses.dataclass(frozen=True)
class _LossConfig:
    type: str
    params: tp.Dict[str, tp.Any]  # kwargs


@dataclasses.dataclass(frozen=True)
class _OptimizerConfig:
    type: str
    params: tp.Dict[str, tp.Any]  # kwargs


@enum.unique
class UpdationLevel(enum.Enum):
    EPOCH = "epoch"
    GSTEP = "gradient_step"
    # TODO: ASTEP = "accumulation_step"

    @classmethod
    def from_str(cls, s: str) -> 'UpdationLevel':
        if s == "epoch":
            return cls.EPOCH
        elif s == "gradient_step":
            return cls.GSTEP
        else:
            raise ValueError(
                f"enum.Enum type UpdationLevel doesn't maintain `{s}` entity."
            )


@dataclasses.dataclass(frozen=True)
class _LRSchedulerConfig:
    use_after: UpdationLevel
    type: str
    params: tp.Dict[str, tp.Any]  # kwargs


@dataclasses.dataclass(frozen=True)
class _HyperParamsConfig:
    loss: _LossConfig
    optimizer: _OptimizerConfig
    lr_scheduler: _LRSchedulerConfig
    total_epoch_amount: int


@dataclasses.dataclass(frozen=True)
class _SubNetOutputConfig:
    sub_net_name: str
    number_of_vectors: int
    inclusion_condition: tp.Callable[[int], bool]


@dataclasses.dataclass(frozen=True)
class _TransformOfY:
    type: str
    params: tp.Dict[str, tp.Any]  # kwargs


@dataclasses.dataclass(frozen=True)
class OneMetricConfig:
    name: str
    function: str
    params: tp.Dict[str, tp.Any]  # kwargs
    target_transform: tp.Optional[_TransformOfY]
    prediction_transform: tp.Optional[_TransformOfY]

    @classmethod
    def from_dict(cls, d: tp.Dict[str, tp.Any]) -> 'OneMetricConfig':
        return cls(
            name=d["name"],
            function=d["function"],
            params=d["params"],
            target_transform=_TransformOfY(**d["target_transform"]),
            prediction_transform=_TransformOfY(**d["prediction_transform"])
        )


@dataclasses.dataclass(frozen=True)
class ManyMetricsConfig:
    main: str
    all: tp.List[OneMetricConfig]


@dataclasses.dataclass(frozen=True)
class LearningConfig:
    dataloaders: _TrainValDataloadersConfig
    hyper_params: _HyperParamsConfig
    device: str  # f.e.: "cuda:0"
    tensorboard_logs: pathlib.PosixPath
    checkpoint_dir: pathlib.PosixPath
    metrics: ManyMetricsConfig

    @classmethod
    def from_dict(cls, d: tp.Dict[str, tp.Any]) -> 'LearningConfig':
        return cls(
            dataloaders=_TrainValDataloadersConfig(
                train=_OneDataloaderConfig(
                    dataset_path=pathlib.Path(
                        d["dataloaders"]["train"]["dataset_path"]
                    ),
                    batch_size=d["dataloaders"]["train"]["batch_size"],
                    shuffle=d["dataloaders"]["train"]["shuffle"],
                    drop_last=d["dataloaders"]["train"]["drop_last"],
                    transforms=list(
                        map(
                            TransformConfig.from_dict,
                            d["dataloaders"]["train"]["transforms"]
                        )
                    )
                ),
                val=_OneDataloaderConfig(
                    dataset_path=pathlib.Path(
                        d["dataloaders"]["val"]["dataset_path"]
                    ),
                    batch_size=d["dataloaders"]["val"]["batch_size"],
                    shuffle=d["dataloaders"]["val"]["shuffle"],
                    drop_last=d["dataloaders"]["val"]["drop_last"],
                    transforms=list(
                        map(
                            TransformConfig.from_dict,
                            d["dataloaders"]["val"]["transforms"]
                        )
                    )
                )
            ),
            hyper_params=_HyperParamsConfig(
                loss=_LossConfig(
                    type=d["hyper_params"]["loss"]["type"],
                    params=d["hyper_params"]["loss"]["params"]
                ),
                optimizer=_OptimizerConfig(
                    type=d["hyper_params"]["optimizer"]["type"],
                    params=d["hyper_params"]["optimizer"]["params"]
                ),
                lr_scheduler=_LRSchedulerConfig(
                    use_after=UpdationLevel.from_str(
                        d["hyper_params"]["lr_scheduler"]["use_after"]
                    ),
                    type=d["hyper_params"]["lr_scheduler"]["type"],
                    params=dict(
                        map(
                            lambda k_v: (
                                k_v[0],
                                eval(k_v[1])
                            ) if k_v[0] == "lr_lambda" else k_v,
                            d["hyper_params"]["lr_scheduler"]["params"].items()  # noqa: E501
                        )
                    )
                ),
                total_epoch_amount=d["hyper_params"]["total_epoch_amount"]
            ),
            device=d["device"],
            tensorboard_logs=pathlib.Path(d["tensorboard_logs"]),
            checkpoint_dir=pathlib.Path(d["checkpoint_dir"]),
            metrics=ManyMetricsConfig(
                main=d["metrics"]["main"],
                all=list(map(OneMetricConfig.from_dict, d["metrics"]["all"]))
            )
        )
