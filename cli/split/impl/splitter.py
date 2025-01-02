import pathlib
import typing as tp

import h5py
import numpy as np

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
        elif self._split_mode == SplitMode.RANDOM:
            return self._random_run()
        else:
            assert False, "Unreachable line!"

    def _dummy_run(self) -> None:
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

    def _random_run(self) -> None:
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
            indexes: np.types.NDArray[int] = np.random.permutation(src_ds_len)
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
                for src_i in indexes[:dst_ds[0]["len"]]:
                    for col in ["input", "output"]:
                        dst0_ds[col][dst0_i] = src_ds[col][src_i]
                    print(dst0_i, end="\r")
                    dst0_i += 1
                print()
                assert dst0_i == dst_ds[0]["len"]
                dst1_i: int = 0
                for src_i in indexes[dst_ds[0]["len"]:]:
                    for col in ["input", "output"]:
                        dst1_ds[col][dst1_i] = src_ds[col][src_i]
                    print(dst1_i, end="\r")
                    dst1_i += 1
                print()
                assert dst1_i == dst_ds[1]["len"]
