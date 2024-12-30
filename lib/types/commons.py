import abc
import dataclasses
import enum
import typing as tp

import torch


@enum.unique
class LearningMode(enum.Enum):
    TRAIN = "train"
    VAL = "val"


@dataclasses.dataclass
class _InputOutputInterface(abc.ABC):
    input: tp.Any  # input for a model
    output: tp.Optional[tp.Any]  # the expected output


class ModelInputOutputPairSample(_InputOutputInterface):
    def __init__(self, input_, output_, /):
        self._input, self._output = input_, output_
        self._validate_types()

    @property
    def input(self) -> tp.Any:
        return self._input

    @property
    def output(self) -> tp.Optional[tp.Any]:
        return self._output

    def _validate_types(self):
        assert type(self.input) is torch.Tensor
        assert (self.output is None) or (type(self.output) is torch.Tensor)


@dataclasses.dataclass  # note that it's mutable!
class RawModelInputOutputPairSample(_InputOutputInterface):
    def finalize(self) -> ModelInputOutputPairSample:
        return ModelInputOutputPairSample(
            torch.from_numpy(self.input),
            torch.from_numpy(self.output),
        )
