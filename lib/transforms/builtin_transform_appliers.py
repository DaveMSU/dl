import typing as tp

import torchvision

from .base import BaseRawModelInputOutputTransform
from lib.types import RawModelInputOutputPairSample


class InputBuiltInTransformApplier(BaseRawModelInputOutputTransform):
    def __init__(self, transform_type: str, params: tp.Dict[str, tp.Any]):
        self._transform = getattr(
            torchvision.transforms,
            transform_type
        )(
            **params
        )

    def __call__(self, sample: RawModelInputOutputPairSample) -> None:
        sample.input = self._transform(sample.input)
