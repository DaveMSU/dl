from ..split_mode import SplitMode
from . import splitter_for_each_mode


def splitter_factory(mode: SplitMode) -> splitter_for_each_mode.BaseSplitter:
    if mode == SplitMode.DUMMY:
        return splitter_for_each_mode.DummySplitter
    elif mode == SplitMode.RANDOM:
        return splitter_for_each_mode.RandomSplitter
    else:
        assert False, "Unreachable line!"
