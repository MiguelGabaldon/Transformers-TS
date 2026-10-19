"""Oracle and training-estimated AR(1) reference forecasts."""

import numpy as np

from .config import DataConfig
from .data import SplitName, split_boundaries, target_bounds
from .metrics import summarize_replica_metrics


def estimate_phi(values: np.ndarray, train_end: int) -> float:
    """Estimate a zero-intercept AR(1) coefficient by pooled least squares."""

    lagged = values[:, : train_end - 1]
    current = values[:, 1:train_end]
    denominator = float(np.square(lagged).sum())
    if denominator == 0:
        raise ValueError("Cannot estimate phi when all lagged values are zero.")
    return float((lagged * current).sum() / denominator)


def evaluate_ar1_baselines(
    values: np.ndarray, config: DataConfig
) -> tuple[float, dict[str, dict[str, dict[str, dict[str, float]]]]]:
    """Evaluate oracle and estimated coefficients on identical target indices."""

    train_end, _ = split_boundaries(config)
    estimated_phi = estimate_phi(values, train_end)
    coefficients = {"ar1_oracle": config.phi, "ar1_estimated": estimated_phi}
    results: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for name, coefficient in coefficients.items():
        results[name] = {}
        for split in ("train", "validation", "test"):
            start, end = target_bounds(config, split)
            targets = values[:, start:end]
            predictions = coefficient * values[:, start - 1 : end - 1]
            results[name][split] = summarize_replica_metrics(targets, predictions)
    return estimated_phi, results
