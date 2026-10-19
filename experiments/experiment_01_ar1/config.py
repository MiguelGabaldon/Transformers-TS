"""Explicit, reproducible configuration for the AR(1) experiment."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DataConfig:
    """Parameters for the data-generating process and forecast windows."""

    phi: float = 0.8
    sigma: float = 1.0
    num_replicas: int = 50
    series_length: int = 2_000
    lookback: int = 24
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    seed: int = 42

    def __post_init__(self) -> None:
        if abs(self.phi) >= 1:
            raise ValueError("Stationary AR(1) generation requires abs(phi) < 1.")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive.")
        if not 0 < self.lookback < self.series_length:
            raise ValueError("lookback must be between 1 and series_length - 1.")
        test_fraction = 1.0 - self.train_fraction - self.validation_fraction
        if min(self.train_fraction, self.validation_fraction, test_fraction) <= 0:
            raise ValueError("Train, validation, and test fractions must be positive.")


@dataclass(frozen=True)
class ModelConfig:
    """Parameters for the deliberately small Transformer encoder."""

    d_model: int = 32
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 64
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        if min(self.d_model, self.nhead, self.num_layers, self.dim_feedforward) <= 0:
            raise ValueError("All model dimensions must be positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization parameters."""

    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 30
    patience: int = 5
    seed: int = 123
    device: str = "auto"

    def __post_init__(self) -> None:
        if min(self.batch_size, self.learning_rate, self.max_epochs, self.patience) <= 0:
            raise ValueError("Training values must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete experiment configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
