from dataclasses import dataclass

@dataclass
class TrainConfig:
    fs_default: int = 125
    input_seconds: int = 10
    sqi_thresh: float = 0.8
    use_sqi: bool = True

    # model
    conv_channels: tuple = (64, 128, 256)
    conv_kernels: tuple = (7, 5, 3)
    transformer_dim: int = 256
    transformer_heads: int = 4
    transformer_layers: int = 2
    dropout: float = 0.1

    # optimization
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 120
    patience: int = 15

    # loss weights (emphasize SBP more than DBP)
    sbp_weight: float = 1.5
    dbp_weight: float = 1.0
