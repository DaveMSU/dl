from lib.types import TokenizerConfig
from lib.types import tokenizers


def tokenizer_factory(
        tokenizer_config: TokenizerConfig
) -> tokenizers.BaseTokenizer:
    if tokenizer_config.type == "CharacterTokenizer":
        return getattr(tokenizers, "CharacterTokenizer")(
            traing_data_limit=tokenizer_config.params["traing_data_limit"]
        )
    else:
        raise TypeError(f"No such class as `{tokenizer_config.type}`")
