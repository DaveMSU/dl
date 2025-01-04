import enum


@enum.unique
class SplitMode(enum.Enum):
    DUMMY = "dummy"
    RANDOM = "random"
