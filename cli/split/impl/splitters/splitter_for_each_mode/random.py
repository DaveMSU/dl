import typing as tp

import h5py
import numpy as np

from .base import BaseSplitter


class RandomSplitter(BaseSplitter):
    def _run_impl(self) -> None:
        with h5py.File(self._src, "r") as src_ds:
            assert {name for name in src_ds} == {"input", "output"}
            src_ds_len: int = len(src_ds["input"])
            assert src_ds_len == len(src_ds["output"])

            dst_ds: tp.Dict[str, tp.Union[int, h5py.Dataset, None]] = {
                0: {"len": None, "input": None, "output": None},
                1: {"len": None, "input": None, "output": None}
            }
            dst_ds[0]["len"] = round(src_ds_len * self._th)
            dst_ds[1]["len"] = src_ds_len - dst_ds[0]["len"]
            _inds: np.types.NDArray[int] = np.random.permutation(src_ds_len)
            indexes: tp.Dict[str, np.types.NDArray[int]] = {
                0: np.sort(_inds[:dst_ds[0]["len"]]),
                1: np.sort(_inds[dst_ds[0]["len"]:])
            }
            del _inds
            with h5py.File(self._dst0, "w") as dst0_ds, \
                    h5py.File(self._dst1, "w") as dst1_ds:
                for col in ["input", "output"]:
                    dst_ds[0][col] = dst0_ds.create_dataset(
                        col,
                        shape=(dst_ds[0]["len"], *src_ds[col].shape[1:]),
                        dtype=src_ds[col].dtype
                    )
                    dst_ds[1][col] = dst1_ds.create_dataset(
                        col,
                        shape=(dst_ds[1]["len"], *src_ds[col].shape[1:]),
                        dtype=src_ds[col].dtype
                    )
                dst0_i: int = 0
                for src_i in indexes[0]:
                    for col in ["input", "output"]:
                        dst0_ds[col][dst0_i] = src_ds[col][src_i]
                    print(dst0_i, end="\r")
                    dst0_i += 1
                print()
                assert dst0_i == dst_ds[0]["len"]
                dst1_i: int = 0
                for src_i in indexes[1]:
                    for col in ["input", "output"]:
                        dst1_ds[col][dst1_i] = src_ds[col][src_i]
                    print(dst1_i, end="\r")
                    dst1_i += 1
                print()
                assert dst1_i == dst_ds[1]["len"]
