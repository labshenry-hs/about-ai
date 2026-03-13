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

    def to_dict(self): return self.__dict__

    @staticmethod
    def small():
        return TransformerConfig(d_model=256, n_heads=4, n_layers=4, d_ff=1024)

    @staticmethod
    def base():
        return TransformerConfig(d_model=512, n_heads=8, n_layers=6, d_ff=2048)

@dataclass
class TrainConfig:
    data_path: str = "data/train.txt"
    output_dir: str = "checkpoints"
    seed: int = 42
    batch_size: int = 32
    max_len: int = 512
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_ratio: float = 0.05
    fp16: bool = True
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 5000

@dataclass
class LLMConfig(TransformerConfig):
    """Config for decoder-only LLM."""
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4
    d_ff: int = 3072
    rope_base: int = 10000
    use_flash_attn: bool = True
    use_moe: bool = False
    n_experts: int = 8
    top_k_experts: int = 2

from dataclasses import field as dc_field

@dataclasses.dataclass
class ExperimentConfig:
    """Full experiment config: model + training + hardware."""
    model: TransformerConfig = dataclasses.field(default_factory=TransformerConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    seed: int = 42
    compile: bool = False
    profile: bool = False
    tags: list = dataclasses.field(default_factory=list)
    def to_dict(self): return dataclasses.asdict(self)
