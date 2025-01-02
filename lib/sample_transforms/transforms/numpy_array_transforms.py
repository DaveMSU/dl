import typing as tp

import numpy as np

from .base import BaseRawModelInputOutputTransform, BaseInputOrOutputTransform
from lib.types import RawModelInputOutputPairSample


class InputOrOutputNDArrayTranspose(BaseInputOrOutputTransform):
    def __init__(self, field: str, args: tp.List[int]):
        super().__init__(field)
        self._args = args

    def __call__(self, sample: RawModelInputOutputPairSample) -> None:
        assert hasattr(sample, "input") and hasattr(sample, "output")
        setattr(
            sample,
            self._field,
            getattr(sample, self._field).transpose(*self._args)
        )


class InputOrOutputNDArrayNormalize(BaseInputOrOutputTransform):
    def __init__(self, field: str, bias: float = 0.0, scale: float = 1.0):
        super().__init__(field)
        self._bias, self._scale = bias, scale

    def __call__(self, sample: RawModelInputOutputPairSample) -> None:
        assert hasattr(sample, "input") and hasattr(sample, "output")
        setattr(
            sample,
            self._field,
            (getattr(sample, self._field) + self._bias) * self._scale
        )
