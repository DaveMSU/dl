from . import transforms
from lib.types import TransformConfig


def transform_factory(
        config: TransformConfig
) -> transforms.BaseRawModelInputOutputTransform:
    return getattr(transforms, config.type)(**config.params)
