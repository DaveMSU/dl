import io
import json
import typing as tp

import numpy as np
from PIL import Image

from .base import BaseInputOrOutputTransform
from lib.types import RawModelInputOutputPairSample


class InputOrOutputBytesToImage(BaseInputOrOutputTransform):
    def __call__(self, sample: RawModelInputOutputPairSample) -> None:
        the_image = Image.open(
            io.BytesIO(getattr(sample, self._field))
        ).convert('RGB')
        setattr(sample, self._field, the_image)


class InputOrOutputBytesToJson(BaseInputOrOutputTransform):
    def __call__(self, sample: RawModelInputOutputPairSample) -> None:
        the_json = json.loads(getattr(sample, self._field))
        setattr(sample, self._field, the_json)


class InputOrOutputImageToNDArray(BaseInputOrOutputTransform):
    def __init__(self, field: str, dtype: str):
        super().__init__(field)
        if (not dtype.startswith('_')) and (dtype in dir(np)):
            self._dtype = dtype
        else:
            raise ValueError(
                f"Something is wrong with the provided dtype, got `{dtype}`"
            )

    def __call__(self, sample: RawModelInputOutputPairSample) -> None:
        if type(getattr(sample, self._field)) is Image.Image:
            the_array = np.array(
                getattr(sample, self._field)
            ).astype(
                getattr(np, self._dtype)
            )
            setattr(sample, self._field, the_array)
        else:
            raise TypeError("The sample has to be an instance of PIL's Image")


class InputOrOutputStrToInt(BaseInputOrOutputTransform):
    def __init__(self, field: str, mapper: tp.Dict[str, int]):
        super().__init__(field)
        self._mapper = dict(**mapper)

    def __call__(self, sample: RawModelInputOutputPairSample) -> None:
        if type(getattr(sample, self._field)) is bytes:
            the_int = self._mapper[getattr(sample, self._field).decode()]
        elif type(getattr(sample, self._field)) is str:
            the_int = self._mapper[getattr(sample, self._field)]
        else:
            raise ValueError(
                f"The str was expected for sample's `{self._field}` as"
                f" a type, but `{type(getattr(sample, self._field))}`"
                " has occured instead"
            )
        setattr(sample, self._field, the_int)


class InputOrOutputIntToOneHot(BaseInputOrOutputTransform):
    def __init__(self, field: str, amount_of_classes: int):
        super().__init__(field)
        assert amount_of_classes >= 2
        self._amount_of_classes = amount_of_classes

    def __call__(self, sample: RawModelInputOutputPairSample) -> None:
        one_hot = np.zeros(self._amount_of_classes, dtype=np.float32)
        one_hot[getattr(sample, self._field)] = 1.0
        setattr(sample, self._field, one_hot)
