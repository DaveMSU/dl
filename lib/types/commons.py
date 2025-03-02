import abc
import enum
import typing as tp

import numpy as np
import torch


class BaseStrictSingleton(abc.ABC):
    _instance: tp.Optional['BaseStrictSingleton'] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is not None:
            raise RuntimeError(
                f"Class `{cls.__name__}` may have only 1 instance!"
            )
        else:
            cls._instance: 'BaseStrictSingleton' = super().__new__(cls)
            return cls._instance

    def __del__(self):
        if self.__class__._instance is self:
            self.__class__._instance = None


@enum.unique
class LearningMode(enum.Enum):
    TRAIN = "train"
    VAL = "val"


class _InputOutputInterface(abc.ABC):
    def __init__(
            self,
            input_: tp.Any,  # input for a model
            output_: tp.Optional[tp.Any],  # the expected output
            /
    ):
        if self._is_none(input_):
            raise ValueError("'input' field isn't optional!")
        else:
            self.input = input_
        if self._is_none(output_):
            self.output = None
        else:
            self.output = output_

    @staticmethod
    def _is_none(obj: tp.Any) -> bool:
        is_nan: bool
        try:
            is_nan = bool(np.isnan(obj))
        except BaseException:
            is_nan = False
        return is_nan or (obj is None)


class ModelInputOutputPairSample(_InputOutputInterface):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_frozen", True)
        self._validate_types()

    def __setattr__(self, field: str, value: tp.Any) -> tp.Any:
        if getattr(self, "_frozen", False):
            raise AttributeError("Can't modify immutable instance")
        super().__setattr__(field, value)

    def _validate_types(self):
        assert type(self.input) is torch.Tensor
        assert (self.output is None) or (type(self.output) is torch.Tensor)


class RawModelInputOutputPairSample(_InputOutputInterface):
    def finalize(self) -> ModelInputOutputPairSample:
        return ModelInputOutputPairSample(
            torch.from_numpy(self.input),
            None if self.output is None else torch.from_numpy(self.output),
        )
