import abc
import pathlib
import typing as tp

from lib.types import BaseStrictSingleton


class _TokenizerInterface(abc.ABC):
    @abc.abstractmethod
    def fit(self, path: pathlib.PosixPath, limit: int) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def encode(self, text: str) -> tp.Sequence[int]:
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, tokens: tp.Sequence[int]) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, path: pathlib.PosixPath) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, path: pathlib.PosixPath) -> None:
        raise NotImplementedError


class BaseTokenizer(BaseStrictSingleton, _TokenizerInterface):
    def __init__(self):
        self._available_to_be_fitted: bool = True

    def fit(self, path: pathlib.PosixPath, limit: int) -> None:
        if not self._available_to_be_fitted:
            raise ValueError("Tokenizer's already fitted")
        else:
            self._fit_impl(path, limit)
            self._available_to_be_fitted &= False

    def save(self, path: pathlib.PosixPath) -> None:
        if self._available_to_be_fitted:
            raise ValueError("Tokenizer hasn't been fitted yet")
        else:
            self._save_impl(path)

    def load(self, path: pathlib.PosixPath) -> None:
        if not self._available_to_be_fitted:
            raise ValueError("Tokenizer's already fitted")
        else:
            self._load_impl(path)
            self._available_to_be_fitted &= False

    @abc.abstractmethod
    def _fit_impl(self, path: pathlib.PosixPath, limit: int) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _save_impl(self, path: pathlib.PosixPath) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _load_impl(self, path: pathlib.PosixPath) -> None:
        raise NotImplementedError
