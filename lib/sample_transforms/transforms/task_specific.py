import dataclasses
import random
import typing as tp

import numpy as np
from PIL import Image

from .base import BaseRawModelInputOutputTransform
from lib.types import FacePoints, Ratio, RawModelInputOutputPairSample


class OutputDictToFacePoints(BaseRawModelInputOutputTransform):
    def __call__(self, sample: RawModelInputOutputPairSample) -> None:
        sample.output = FacePoints(**sample.output)


class FacePointsToNDArray(BaseRawModelInputOutputTransform):
    def __init__(self, dtype: str):
        if (not dtype.startswith('_')) and (dtype in dir(np)):
            self._dtype = dtype
        else:
            raise ValueError(
                f"Something is wrong with the provided dtype, got `{dtype}`"
            )

    def __call__(self, sample: RawModelInputOutputPairSample) -> None:
        sample.output = np.array(
            [
                getattr(sample.output, k) for k in [
                    "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4",
                    "x5", "y5", "x6", "y6", "x7", "y7", "x8", "y8",
                    "x9", "y9",
                    "x10", "y10", "x11", "y11", "x12", "y12",
                    "x13", "y13", "x14", "y14"
                ]
            ],
            dtype=self._dtype
        )


class FaceAndAbsPointsResize(BaseRawModelInputOutputTransform):
    def __init__(self, size: tp.List[int]):
        assert len(size) == 2
        assert type(size[0]) is int
        assert type(size[1]) is int
        self._new_size: tp.Tuple[int, int] = tuple(size)

    def __call__(self, sample: RawModelInputOutputPairSample) -> None:
        old_size: tp.Tuple[int, int]
        old_size, sample.input = sample.input.size, sample.input.resize(
            self._new_size,
            Image.Resampling.NEAREST
        )
        assert sample.input.size == self._new_size
        for field in dataclasses.fields(sample.output):
            coord: float = getattr(sample.output, field.name)
            _id = {'x': 0, 'y': 1}[field.name[0]]
            setattr(
                sample.output,
                field.name,
                coord / old_size[_id] * self._new_size[_id]
            )


class FaceAndAbsPointsHorizontalRandomFlip(BaseRawModelInputOutputTransform):
    def __init__(self, probability: float):
        self._p = Ratio(probability)

    def __call__(self, sample: RawModelInputOutputPairSample) -> None:
        if (random.random() < self._p) or (self._p == 1.0):
            sample.input = sample.input.transpose(Image.FLIP_LEFT_RIGHT)
            sample.output = FacePoints(
                x1=sample.input.width - sample.output.x4, y1=sample.output.y4,
                x2=sample.input.width - sample.output.x3, y2=sample.output.y3,
                x3=sample.input.width - sample.output.x2, y3=sample.output.y2,
                x4=sample.input.width - sample.output.x1, y4=sample.output.y1,
                x5=sample.input.width - sample.output.x10, y5=sample.output.y10,  # noqa: E501
                x6=sample.input.width - sample.output.x9, y6=sample.output.y9,
                x7=sample.input.width - sample.output.x8, y7=sample.output.y8,
                x8=sample.input.width - sample.output.x7, y8=sample.output.y7,
                x9=sample.input.width - sample.output.x6, y9=sample.output.y6,
                x10=sample.input.width - sample.output.x5, y10=sample.output.y5,  # noqa: E501
                x11=sample.input.width - sample.output.x11, y11=sample.output.y11,  # noqa: E501
                x12=sample.input.width - sample.output.x14, y12=sample.output.y14,  # noqa: E501
                x13=sample.input.width - sample.output.x13, y13=sample.output.y13,  # noqa: E501
                x14=sample.input.width - sample.output.x12, y14=sample.output.y12  # noqa: E501
            )


class MakeAbsolutePointCoordsRelative(BaseRawModelInputOutputTransform):  # noqa: E501
    def __call__(self, sample: RawModelInputOutputPairSample) -> None:
        for field in dataclasses.fields(sample.output):
            coord: float = getattr(sample.output, field.name)
            setattr(
                sample.output,
                field.name,
                coord / sample.input.size[{'x': 0, 'y': 1}[field.name[0]]]
            )
