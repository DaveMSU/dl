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
            transforms: tp.Tuple[BaseRawModelInputOutputTransform, ...]
    ):
        super().__init__(src_dataset_path, dst_dataset_path)
        self._transforms = transforms

    def run(self) -> None:
        with h5py.File(self._src, "r") as src_ds:
            assert {name for name in src_ds} == {"input", "output"}
            src_ds_len: int = len(src_ds["input"])
            assert src_ds_len == len(src_ds["output"])

            self._ref_sample = self._produce_wrangled_sample(src_ds, 0)
            self._validate(self._ref_sample)
            with h5py.File(self._dst, "w") as dst_ds:
                dst_ds_cols: tp.Dict[str, h5py.Dataset] = dict()
                for col in ["input", "output"]:
                    dst_ds_cols[col] = dst_ds.create_dataset(
                        col,
                        shape=(
                            src_ds_len,
                            *getattr(self._ref_sample, col).shape
                        ),
                        dtype=getattr(self._ref_sample, col).dtype
                    )
                for i in range(src_ds_len):
                    sample = self._produce_wrangled_sample(src_ds, i)
                    self._validate(sample)
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

    def _validate(self, sample: RawModelInputOutputPairSample) -> None:
        for field in ["input", "output"]:
            if type(sf := getattr(sample, field)) is not np.ndarray:
                raise ValueError(
                    f"`{field}` of the sample must be np.ndarray,"
                    f" but got `{type(sf)}`"
                )
            if (rfs := getattr(self._ref_sample, field).shape) != sf.shape:
                raise ValueError(
                    "Shapes of the sample and reference sample mush be equal,"
                    f" but got `{sf.shape}` and `{rfs}` respectively"
                )
