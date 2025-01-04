import abc
import pathlib
import typing as tp

from lib.types import Ratio


class BaseSplitter(abc.ABC):
    _instance: tp.Optional['BaseSplitter'] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is not None:
            raise RuntimeError(
                f"Class `{cls.__name__}` may have only 1 instance!"
            )
        else:
            cls._instance: 'BaseSplitter' = super().__new__(cls)
            return cls._instance

    def __init__(
            self,
            src: pathlib.PosixPath,
            dst0: pathlib.PosixPath,
            dst1: pathlib.PosixPath,
            threshold: Ratio,
    ):
        self._src = src
        self._dst0 = dst0
        self._dst1 = dst1
        self._th = threshold
        self._has_run_been_invoked: bool = False

    def run(self) -> None:
        if self._has_run_been_invoked:
            raise RuntimeError(
                "Method '.run()' may not be invoked more than once!"
            )
        else:
            self._run_impl()
            self._has_run_been_invoked |= True

    @abc.abstractmethod
    def _run_impl(self) -> None:
        raise NotImplementedError
