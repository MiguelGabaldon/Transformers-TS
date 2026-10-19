"""AR(1) generation, chronological splitting, and supervised windows."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch.utils.data import TensorDataset

from .config import DataConfig

SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class Standardizer:
    """Scalar standardization fitted only on training observations."""

    mean: float
    std: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


@dataclass(frozen=True)
class WindowedSplit:
    """Flattened windows with enough metadata for per-replica evaluation."""

    inputs: torch.Tensor
    targets: torch.Tensor
    num_replicas: int
    targets_per_replica: int

    @property
    def dataset(self) -> TensorDataset:
        return TensorDataset(self.inputs, self.targets)


def generate_ar1(config: DataConfig) -> np.ndarray:
    """Generate independent stationary AR(1) replicas with shape ``[R, T]``.

    The initial observation is drawn from the exact stationary distribution,
    N(0, sigma^2 / (1 - phi^2)), so no burn-in is necessary.
    """

    rng = np.random.default_rng(config.seed)
    values = np.empty((config.num_replicas, config.series_length), dtype=np.float32)
    stationary_std = config.sigma / np.sqrt(1.0 - config.phi**2)
    values[:, 0] = rng.normal(0.0, stationary_std, config.num_replicas)
    innovations = rng.normal(
        0.0,
        config.sigma,
        size=(config.num_replicas, config.series_length - 1),
    )
    for time_index in range(1, config.series_length):
        values[:, time_index] = (
            config.phi * values[:, time_index - 1] + innovations[:, time_index - 1]
        )
    return values


def split_boundaries(config: DataConfig) -> tuple[int, int]:
    """Return the exclusive train and validation endpoints."""

    train_end = int(config.series_length * config.train_fraction)
    validation_end = train_end + int(
        config.series_length * config.validation_fraction
    )
    if config.lookback >= train_end:
        raise ValueError("lookback must be shorter than the training split.")
    return train_end, validation_end


def fit_training_standardizer(values: np.ndarray, train_end: int) -> Standardizer:
    """Fit normalization using raw training observations only."""

    training_values = values[:, :train_end]
    std = float(training_values.std())
    if std == 0:
        raise ValueError("Cannot standardize constant training data.")
    return Standardizer(mean=float(training_values.mean()), std=std)


def target_bounds(config: DataConfig, split: SplitName) -> tuple[int, int]:
    """Return inclusive/exclusive target indices for a chronological split."""

    train_end, validation_end = split_boundaries(config)
    if split == "train":
        return config.lookback, train_end
    if split == "validation":
        return train_end, validation_end
    if split == "test":
        return validation_end, config.series_length
    raise ValueError(f"Unknown split: {split}")


def make_windowed_split(
    values: np.ndarray,
    config: DataConfig,
    split: SplitName,
    standardizer: Standardizer,
) -> WindowedSplit:
    """Create ``x[t-L:t] -> x[t]`` examples without target leakage.

    Validation and test windows may contain observations before their target
    split. Those observations are legitimate forecast-time history; every
    target itself remains strictly inside the requested split.
    """

    if values.shape != (config.num_replicas, config.series_length):
        raise ValueError(
            "values must have shape "
            f"({config.num_replicas}, {config.series_length}), got {values.shape}."
        )
    start, end = target_bounds(config, split)
    normalized = standardizer.transform(values)
    windows = np.lib.stride_tricks.sliding_window_view(
        normalized, window_shape=config.lookback, axis=1
    )
    selected_windows = windows[:, start - config.lookback : end - config.lookback]
    selected_targets = normalized[:, start:end]
    targets_per_replica = end - start

    # Copy removes read-only/strided NumPy layouts before conversion to Torch.
    inputs = torch.from_numpy(
        selected_windows.reshape(-1, config.lookback).copy()
    ).float()
    targets = torch.from_numpy(selected_targets.reshape(-1).copy()).float()
    return WindowedSplit(
        inputs=inputs,
        targets=targets,
        num_replicas=config.num_replicas,
        targets_per_replica=targets_per_replica,
    )


def prepare_data(
    config: DataConfig,
) -> tuple[np.ndarray, Standardizer, dict[SplitName, WindowedSplit]]:
    """Generate data and construct all three chronological datasets."""

    values = generate_ar1(config)
    train_end, _ = split_boundaries(config)
    standardizer = fit_training_standardizer(values, train_end)
    splits = {
        split: make_windowed_split(values, config, split, standardizer)
        for split in ("train", "validation", "test")
    }
    return values, standardizer, splits
