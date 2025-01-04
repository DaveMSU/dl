import typing as tp

import h5py

from .base import BaseSplitter
from lib.types import Ratio


class DummySplitter(BaseSplitter):
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
                for i in range(dst_ds[0]["len"]):
                    for col in ["input", "output"]:
                        dst0_ds[col][i] = src_ds[col][i]
                    print(i, end="\r")
                print()
                for i in range(dst_ds[1]["len"]):
                    for col in ["input", "output"]:
                        dst1_ds[col][i] = src_ds[col][i + dst_ds[0]["len"]]
                    print(i, end="\r")
                print()
