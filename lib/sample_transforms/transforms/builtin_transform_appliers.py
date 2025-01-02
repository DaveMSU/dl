import typing as tp

import torchvision

from .base import BaseInputOrOutputTransform
from lib.types import RawModelInputOutputPairSample


class InputOrOutputBuiltInTransformApplier(BaseInputOrOutputTransform):
    def __init__(
            self,
            field: str,
            transform_type: str,
            params: tp.Dict[str, tp.Any]
    ):
        super().__init__(field)
        self._transform = getattr(
            torchvision.transforms,
            transform_type
        )(
            **params
        )

    def __call__(self, sample: RawModelInputOutputPairSample) -> None:
        setattr(
            sample,
            self._field,
            self._transform(getattr(sample, self._field))
        )
