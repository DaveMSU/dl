import io
import json
import pathlib

import numpy as np
import torch
from PIL import Image

from .base import BaseRawModelInputOutputPairSample
from ..commons import ModelInputOutputPairSample


# TODO: use meta-class?
class ImageAndLabel(BaseRawModelInputOutputPairSample):
    @classmethod
    def create_instance(
            cls,
            input_: np.typing.NDArray[np.uint8],  # binary data
            output_: str  # label
    ) -> 'ImageAndLabel':
        assert type(output_) is str
        return cls(
            input=Image.open(io.BytesIO(input_)).convert('RGB'),
            output=output_
        )

    def wrangle_itself(self) -> ModelInputOutputPairSample:
        assert type(self.output) is np.ndarray, type(self.output)
        assert (self.output.ndim == 1) and (self.output.shape[0] > 1)
        assert self.output.dtype == np.float32
        return ModelInputOutputPairSample(
            torch.from_numpy(
                np.array(self.input).astype(np.float32).transpose(2, 0, 1)
            ) / 255.,
            torch.from_numpy(self.output)  # TODO: is this the right shape?
        )
