from lib.types import TokenizerConfig
from lib.types import tokenizers


def tokenizer_factory(
        tokenizer_config: TokenizerConfig
) -> tokenizers.BaseTokenizer:
    if tokenizer_config.type == "CharacterTokenizer":
        return getattr(tokenizers, "CharacterTokenizer")(
            **tokenizer_config.params  # should be empty dict
        )
    else:
        raise TypeError(f"No such class as `{tokenizer_config.type}`")
