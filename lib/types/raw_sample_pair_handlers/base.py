import abc
import dataclasses
import pathlib
import typing as tp

from ..commons import ModelInputOutputPairSample


@dataclasses.dataclass  # note that it's mutable!
class BaseRawModelInputOutputPairSample(abc.ABC):
    input: tp.Any  # TODO: better typing
    output: tp.Any  # TODO: better typing

    @classmethod
    @abc.abstractmethod
    def create_instance(
            cls,
            input_: tp.Any,
            output_: tp.Any,
            /
    ) -> 'BaseRawModelInputOutputPairSample':
        raise NotImplementedError

    @abc.abstractmethod
    def wrangle_itself(
            self
    ) -> ModelInputOutputPairSample:
        raise NotImplementedError
