import abc

from lib.types import RawModelInputOutputPairSample


class BaseRawModelInputOutputTransform(abc.ABC):
    @abc.abstractmethod
    def __call__(self, sample: RawModelInputOutputPairSample) -> None:
        raise NotImplementedError


class BaseInputOrOutputTransform(BaseRawModelInputOutputTransform):
    def __init__(self, field: str):
        if field not in ["input", "output"]:
            raise ValueError(
                f"field must be 'input' or 'output', got `{field}`"
            )
        else:
            self._field = field
