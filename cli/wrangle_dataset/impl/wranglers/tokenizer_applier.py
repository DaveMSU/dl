import pathlib

import h5py
import numpy as np

from .base import BaseWrangler
from lib.types import (
    RawModelInputOutputPairSample,
    tokenizers,
)


class TokenizerApplier(BaseWrangler):
    def __init__(
            self,
            src_dataset_path: pathlib.PosixPath,
            dst_dataset_path: pathlib.PosixPath,
            tokenizer: tokenizers.BaseTokenizer
    ):
        super().__init__(src_dataset_path, dst_dataset_path)
        self._tokenizer = tokenizer

    def run(self) -> None:
        with h5py.File(self._src, "r") as src_ds:
            assert {name for name in src_ds} == {"input", "output"}
            src_ds_len: int = len(src_ds["input"])
            assert src_ds_len == len(src_ds["output"])

            self._ref_sample = self._produce_wrangled_sample(src_ds, 0)
            self._validate(self._ref_sample)
            with h5py.File(self._dst, "w") as dst_ds:
                dst_ds_column = dict()
                for col in ["input", "output"]:
                    dst_ds_column[col] = dst_ds.create_dataset(
                        col,
                        shape=(src_ds_len,),
                        dtype=h5py.vlen_dtype(np.int16)
                    )
                for i in range(src_ds_len):
                    sample = self._produce_wrangled_sample(src_ds, i)
                    self._validate(sample)
                    for col in ["input", "output"]:
                        dst_ds_column[col][i] = getattr(sample, col)
                    print(i, end='\r')

    def _produce_wrangled_sample(
            self,
            ds: h5py.File,
            index: int
    ) -> RawModelInputOutputPairSample:
        sample = RawModelInputOutputPairSample(
            ds["input"][index],
            ds["output"][index]
        )
        sample.input = np.array(
            self._tokenizer.encode(sample.input.decode()),
            dtype=np.int16  # TODO: invoke len from _TokenizerInterface
        )
        sample.output = np.array(
            self._tokenizer.encode(sample.output.decode()),
            dtype=np.int16  # TODO: invoke len from _TokenizerInterface
        )
        return sample

    def _validate(self, sample: RawModelInputOutputPairSample) -> None:
        for field in ["input", "output"]:
            if type(sf := getattr(sample, field)) is not np.ndarray:
                raise ValueError(
                    f"`{field}` of the sample must be np.ndarray,"
                    f" but got `{type(sf)}`"
                )
