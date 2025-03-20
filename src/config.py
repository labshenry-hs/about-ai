from dataclasses import dataclass

@dataclass
class TransformerConfig:
    vocab_size: int = 10000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_len: int = 512
    batch_size: int = 32
    lr: float = 1e-4
    warmup_steps: int = 4000
    epochs: int = 10

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
