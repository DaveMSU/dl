import typing as tp

import numpy as np

from .base import BaseInputOrOutputTransform
from lib.types import RawModelInputOutputPairSample


class InputOrOutput1DArrayAddPadding(BaseInputOrOutputTransform):
    def __init__(
            self,
            field: str,
            pad_sequences: tp.List[
                tp.List[tp.Union[int, float]]
            ]
    ):
        super().__init__(field)
        self._pad_sequences = pad_sequences

    def __call__(self, sample: RawModelInputOutputPairSample) -> None:
        assert hasattr(sample, "input") and hasattr(sample, "output")
        stored: np.types.NDArray = getattr(sample, self._field)  # 1DArray
        if stored.ndim != 1:
            raise ValueError(
                f"1D Array is expected, `{stored.ndim}D` occured"
            )
        for seq in self._pad_sequences:
            for value in seq:
                if not isinstance(value, type(stored[0].item())):
                    raise TypeError(
                        f"Stored value has type `{type(value)}`,",
                        f" but sample's `{type(stored[0])}` after converting is",
                        f" `{type(stored[0].item())}`"
                    )
        else:
            padded = stored
            for side, seq in zip(["left", "right"], self._pad_sequences):
                for value in seq[::-1 if side == "left" else 1]:
                    padded: np.types.NDArray = np.pad(
                        array=padded,
                        pad_width=(1, 0) if side == "left" else (0, 1),
                        mode="constant",
                        constant_values=value
                    )
            setattr(sample, self._field, padded)


class InputOrOutput1DArrayPadToFinalLength(BaseInputOrOutputTransform):
    def __init__(
            self,
            field: str,
            side: str,
            final_length: tp.Optional[int],
            value: tp.Union[int, float]
    ):
        super().__init__(field)
        if side not in ["left", "right"]:
            raise ValueError("side must be 'left' or 'right'")
        else:
            self._side = side
        self._final_length = final_length
        self._value = value

    def __call__(self, sample: RawModelInputOutputPairSample) -> None:
        assert hasattr(sample, "input") and hasattr(sample, "output")
        stored: np.types.NDArray = getattr(sample, self._field)  # 1DArray
        if stored.ndim != 1:
            raise ValueError(
                f"1D Array is expected, `{stored.ndim}` occured"
            )
        elif not isinstance(self._value, type(stored[0].item())):
            raise TypeError(
                f"Stored value has type `{type(self._value)}`,",
                f" but sample's `{type(stored[0])}` after converting is",
                f" `{type(stored[0].item())}`"
            )
        else:
            to_pad: int = max(0, self._final_length - stored.shape[0])
            padded: np.types.NDArray = np.pad(
                getattr(sample, self._field),
                pad_width=(to_pad, 0) if self._side == "left" else (0, to_pad),
                mode="constant",
                constant_values=self._value
            )
            if (self._side == "right") or (self._final_length == 0):
                setattr(sample, self._field, padded[:self._final_length])
            else:
                setattr(sample, self._field, padded[-self._final_length:])


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
