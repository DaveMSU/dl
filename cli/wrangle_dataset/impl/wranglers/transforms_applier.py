import dataclasses
import pathlib
import typing as tp

import h5py
import numpy as np

from .base import BaseWrangler
from lib.sample_transforms.transforms import BaseRawModelInputOutputTransform
from lib.types import RawModelInputOutputPairSample


class TransformsApplier(BaseWrangler):
    def __init__(
            self,
            src_dataset_path: pathlib.PosixPath,
            dst_dataset_path: pathlib.PosixPath,
            transforms: tp.Tuple[BaseRawModelInputOutputTransform]
    ):
        super().__init__(src_dataset_path, dst_dataset_path)
        self._transforms = transforms

    def run(self) -> None:
        with h5py.File(self._src, "r") as src_ds:
            assert {name for name in src_ds} == {"input", "output"}
            src_ds_len: int = len(src_ds["input"])
            assert src_ds_len == len(src_ds["output"])

            ref_sample = self._produce_wrangled_sample(src_ds, 0)
            assert type(ref_sample.input) is np.ndarray  # TODO: use exception
            assert type(ref_sample.output) is np.ndarray  # TODO: -=-
            with h5py.File(self._dst, "w") as dst_ds:
                dst_ds_cols: tp.Dict[str, h5py.Dataset] = dict()
                for col in ["input", "output"]:
                    dst_ds_cols[col] = dst_ds.create_dataset(
                        col,
                        shape=(src_ds_len, *getattr(ref_sample, col).shape),
                        dtype=getattr(ref_sample, col).dtype
                    )
                for i in range(src_ds_len):
                    sample = self._produce_wrangled_sample(src_ds, i)
                    assert ref_sample.input.shape == sample.input.shape
                    assert ref_sample.output.shape == sample.output.shape
                    for col in ["input", "output"]:
                        dst_ds[col][i] = getattr(sample, col)
                    print(i, end="\r")
                print()

    def _produce_wrangled_sample(
            self,
            ds: h5py.File,
            index: int
    ) -> RawModelInputOutputPairSample:
        sample = RawModelInputOutputPairSample(
            ds["input"][index],
            ds["output"][index]
        )
        for transform in self._transforms:
            transform(sample)
        return sample
