"""Regression metrics reported globally and across independent replicas."""

import numpy as np


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Compute RMSE and MAE for equally shaped arrays."""

    if targets.shape != predictions.shape:
        raise ValueError("targets and predictions must have the same shape.")
    errors = predictions - targets
    return {
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mae": float(np.mean(np.abs(errors))),
    }


def summarize_replica_metrics(
    targets: np.ndarray, predictions: np.ndarray
) -> dict[str, dict[str, float]]:
    """Summarize errors for arrays shaped ``[replica, forecast]``."""

    if targets.ndim != 2 or targets.shape != predictions.shape:
        raise ValueError("Expected matching [replica, forecast] arrays.")
    errors = predictions - targets
    replica_rmse = np.sqrt(np.mean(np.square(errors), axis=1))
    replica_mae = np.mean(np.abs(errors), axis=1)
    # A single-replica smoke run has no sample standard deviation; report zero
    # rather than emitting NaN into the JSON results.
    degrees_of_freedom = 1 if targets.shape[0] > 1 else 0
    return {
        "aggregate": regression_metrics(targets, predictions),
        "across_replicas": {
            "rmse_mean": float(replica_rmse.mean()),
            "rmse_std": float(replica_rmse.std(ddof=degrees_of_freedom)),
            "mae_mean": float(replica_mae.mean()),
            "mae_std": float(replica_mae.std(ddof=degrees_of_freedom)),
        },
    }
