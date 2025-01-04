import h5py
import pathlib
import typing as tp

import torch

from lib.sample_transforms.transforms import BaseRawModelInputOutputTransform
from lib.types import (
    ModelInputOutputPairSample,
    RawModelInputOutputPairSample,
)


class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            hdf5_file_path: pathlib.PosixPath,
            transforms: tp.Tuple[BaseRawModelInputOutputTransform, ...]
     ):
        self._transforms = transforms
        self._hdf5_file = h5py.File(hdf5_file_path, "r")
        self._hdf5_ds_input = self._hdf5_file["input"]
        self._hdf5_ds_output = self._hdf5_file["output"]

    def __len__(self):
        assert len(self._hdf5_ds_input) == len(self._hdf5_ds_output)
        return len(self._hdf5_ds_input)

    def __getitem__(self, index: int) -> tp.Tuple[
            torch.Tensor,
            tp.Optional[torch.Tensor]
    ]:
        sample = RawModelInputOutputPairSample(
            self._hdf5_ds_input[index],
            self._hdf5_ds_output[index]
        )
        for transform in self._transforms:
            transform(sample)
        sample: ModelInputOutputPairSample = sample.finalize()
        return sample.input, sample.output

    def __del__(self):
        self._hdf5_file.close()
