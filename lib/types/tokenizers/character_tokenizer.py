import json
import pathlib
import typing as tp
from collections import Counter
from itertools import chain

import h5py
import numpy as np

from .base import BaseTokenizer


_UNICODE_REPLACEMENT_CHARACTER = chr(65533)


class CharacterTokenizer(BaseTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._char_to_token: tp.Dict[str, int] = dict()
        self._token_to_char: tp.Dict[int, str] = dict()

    def encode(self, text: str) -> tp.List[int]:
        assert type(text) is str
        return list(
            map(lambda char: self._char_to_token.get(char, -1), text)
        )

    def decode(self, tokens: tp.Sequence[int]) -> str:
        return ''.join(
            map(
                lambda token: self._token_to_char.get(
                    token,
                    _UNICODE_REPLACEMENT_CHARACTER
                ),
                tokens
            )
        )

    def _fit_impl(self, path: pathlib.PosixPath, limit: int) -> None:
        with h5py.File(path, "r") as src_ds:
            assert {name for name in src_ds} == {"input", "output"}
            src_ds_len: int = len(src_ds["input"])
            assert src_ds_len == len(src_ds["output"])

            indexes: np.types.NDArray[int] = np.sort(
                np.random.permutation(src_ds_len)[:limit]
            )

            def _char_generator(indx: int):
                print(indx, end="\r")
                for c in chain(
                        src_ds["input"][indx].decode(),
                        src_ds["output"][indx].decode()
                ):
                    yield c
            counter = Counter(
                chain(c for i in indexes for c in _char_generator(i))
            )
        self._char_to_token = {
            char: i for i, (char, cnt) in enumerate(counter.most_common())
        }
        self._token_to_char = {t: c for c, t in self._char_to_token.items()}

    def _save_impl(self, path: pathlib.PosixPath) -> None:
        with path.open("w") as f:
            json.dump(self._char_to_token, f)

    def _load_impl(self, path: pathlib.PosixPath) -> None:
        with path.open("r") as f:
            self._char_to_token = json.load(f)
        self._token_to_char = {t: c for c, t in self._char_to_token.items()}
