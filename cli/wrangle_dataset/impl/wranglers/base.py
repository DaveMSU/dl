import abc
import pathlib


from lib.types import BaseStrictSingleton


class BaseWrangler(BaseStrictSingleton):
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
