import abc
import dataclasses
import pathlib
import typing as tp

import h5py

from lib.sample_transforms.transforms import BaseRawModelInputOutputTransform
from lib.types import RawModelInputOutputPairSample


class BaseWrangler(abc.ABC):
    def __init__(
        self,
        src_dataset_path: pathlib.PosixPath,
        dst_dataset_path: pathlib.PosixPath
    ):
        self._src = src_dataset_path
        self._dst = dst_dataset_path

    @abc.abstractmethod
    def run(self) -> None:
        raise NotImplementedError
