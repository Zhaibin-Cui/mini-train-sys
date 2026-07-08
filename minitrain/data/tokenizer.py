
class Tokenizer:
    """Minimal tokenizer interface.

    Start with an existing tokenizer wrapper, then keep this stable so training
    code does not depend on a specific tokenizer library.
    """

    vocab_size: int

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, tokens: list[int]) -> str:
        raise NotImplementedError

