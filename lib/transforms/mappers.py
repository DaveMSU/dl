import io
import typing as tp

import numpy as np
from PIL import Image

from .base import BaseRawModelInputOutputTransform
from lib.types import RawModelInputOutputPairSample


class InputBytesToImage(BaseRawModelInputOutputTransform):
    def __call__(self, sample: RawModelInputOutputPairSample) -> None:
        sample.input = Image.open(io.BytesIO(sample.input)).convert('RGB')


class InputImageToNDArray(BaseRawModelInputOutputTransform):
    def __init__(self, dtype: str):
        if (not dtype.startswith('_')) and (dtype in dir(np)):
            self._dtype = dtype
        else:
            raise ValueError(
                f"Something is wrong with the provided dtype, got `{dtype}`"
            )

    def __call__(self, sample: RawModelInputOutputPairSample) -> None:
        if type(sample.input) is Image.Image:
            sample.input = np.array(sample.input).astype(
                getattr(np, self._dtype)
            )
        else:
            raise TypeError("The sample has to be an instance of PIL's Image")


class OutputStrToInt(BaseRawModelInputOutputTransform):
    def __init__(self, mapper: tp.Dict[str, int]):
        self._mapper = dict(**mapper)

    def __call__(self, sample: RawModelInputOutputPairSample) -> None:
        if type(sample.output) is bytes:
            sample.output = self._mapper[sample.output.decode()]
        elif type(sample.output) is str:
            sample.output = self._mapper[sample.output]
        else:
            raise ValueError(
                "The str was expected for sample's output as a type,"
                f" but `{type(sample.output)}` has occured instead"
            )


class OutputIntToOneHot(BaseRawModelInputOutputTransform):
    def __init__(self, amount_of_classes: int):
        assert amount_of_classes >= 2
        self._amount_of_classes = amount_of_classes

    def __call__(self, sample: RawModelInputOutputPairSample) -> None:
        one_hot = np.zeros(self._amount_of_classes, dtype=np.float32)
        one_hot[sample.output] = 1.0
        sample.output = one_hot
