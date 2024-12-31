import pathlib

import h5py

from .split_mode import SplitMode
from lib.types import Ratio


class Splitter:
    def __init__(
            self,
            src: pathlib.PosixPath,
            dst0: pathlib.PosixPath,
            dst1: pathlib.PosixPath,
            threshold: Ratio,
            split_mode: SplitMode,
    ):
        self._src = src
        self._dst0 = dst0
        self._dst1 = dst1
        self._th = threshold
        self._split_mode = split_mode

    def run(self) -> None:
        if self._split_mode == SplitMode.DUMMY:
            return self._dummy_run()
        else:
            assert False, "Unreachable line!"

    def _dummy_run(self) -> None:
        with h5py.File(self._src, "r") as src_h5_ds:
            _any_name: str = next(iter(src_h5_ds))
            abs_dst0_len: int = round(len(src_h5_ds[_any_name]) * self._th)
            abs_dst1_len: int = len(src_h5_ds[_any_name]) - abs_dst0_len
            with h5py.File(self._dst0, "w") as dst0_h5_ds, \
                    h5py.File(self._dst1, "w") as dst1_h5_ds:
                for name in src_h5_ds:
                    print(name)
                    src_h5_ds_col = src_h5_ds[name]
                    _shape = list(src_h5_ds_col.shape)
                    dst0_h5_ds_col = dst0_h5_ds.create_dataset(
                        name,
                        shape=(abs_dst0_len, *_shape[1:]),
                        dtype=src_h5_ds_col.dtype
                    )
                    dst1_h5_ds_col = dst1_h5_ds.create_dataset(
                        name,
                        shape=(abs_dst1_len, *_shape[1:]),
                        dtype=src_h5_ds_col.dtype
                    )
                    for i in range(len(src_h5_ds_col)):
                        if i < abs_dst0_len:
                            dst0_h5_ds_col[i] = src_h5_ds_col[i]
                        else:
                            dst1_h5_ds_col[i - abs_dst0_len] = src_h5_ds_col[i]  # noqa: E503
                        print(i, end="\r")
                    print()
