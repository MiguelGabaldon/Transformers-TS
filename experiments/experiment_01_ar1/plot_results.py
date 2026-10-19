"""Create visual validation figures for a completed AR(1) experiment run."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import DataConfig, ModelConfig
from .data import Standardizer, make_windowed_split
from .model import ARTransformer
from .train import predict


@dataclass(frozen=True)
class RunData:
    """Saved data, configuration, and predictions needed by all figures."""

    values: np.ndarray
    train_end: int
    validation_end: int
    data_config: DataConfig
    estimated_phi: float
    targets: np.ndarray
    transformer_predictions: np.ndarray


def load_run(run_directory: Path) -> RunData:
    """Load a completed run and reproduce its Transformer test predictions."""

    required = ("config.json", "metrics.json", "data.npz", "model.pt")
    missing = [name for name in required if not (run_directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Run is missing required files: {', '.join(missing)}")

    config_payload = json.loads(
        (run_directory / "config.json").read_text(encoding="utf-8")
    )
    metrics_payload = json.loads(
        (run_directory / "metrics.json").read_text(encoding="utf-8")
    )
    data_config = DataConfig(**config_payload["data"])
    model_config = ModelConfig(**config_payload["model"])
    batch_size = int(config_payload["training"]["batch_size"])

    with np.load(run_directory / "data.npz") as saved_data:
        values = saved_data["values"].copy()
        train_end = int(saved_data["train_end"])
        validation_end = int(saved_data["validation_end"])

    checkpoint = torch.load(
        run_directory / "model.pt", map_location="cpu", weights_only=True
    )
    standardizer = Standardizer(**checkpoint["standardizer"])
    model = ARTransformer(data_config.lookback, model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_split = make_windowed_split(values, data_config, "test", standardizer)
    targets, predictions = predict(
        model, test_split, standardizer, batch_size, torch.device("cpu")
    )
    return RunData(
        values=values,
        train_end=train_end,
        validation_end=validation_end,
        data_config=data_config,
        estimated_phi=float(metrics_payload["estimated_phi"]),
        targets=targets,
        transformer_predictions=predictions,
    )


def save_figure(
    figure: plt.Figure, output_path: Path, overwrite: bool = False
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {output_path}. Pass --overwrite to replace it."
        )
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_generated_series(run: RunData, num_series: int) -> plt.Figure:
    """Plot several full replicas and their chronological split boundaries."""

    num_series = min(num_series, run.values.shape[0])
    figure, axes = plt.subplots(
        num_series, 1, figsize=(11, 2.1 * num_series), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    time = np.arange(run.values.shape[1])
    for replica, axis in enumerate(axes):
        axis.axvspan(0, run.train_end, color="#4C78A8", alpha=0.06)
        axis.axvspan(
            run.train_end, run.validation_end, color="#F58518", alpha=0.08
        )
        axis.axvspan(
            run.validation_end, run.values.shape[1], color="#54A24B", alpha=0.08
        )
        axis.plot(time, run.values[replica], color="#303030", linewidth=0.7)
        axis.axvline(run.train_end, color="#F58518", linewidth=1, linestyle="--")
        axis.axvline(
            run.validation_end, color="#54A24B", linewidth=1, linestyle="--"
        )
        axis.set_ylabel(f"Replica {replica + 1}")
        axis.grid(alpha=0.15)
    axes[0].set_title(
        "Generated stationary AR(1) replicas "
        f"($\\phi={run.data_config.phi}$, $\\sigma={run.data_config.sigma}$)"
    )
    axes[-1].set_xlabel("Time index")
    figure.text(
        0.5,
        0.01,
        "Blue shading: train   Orange: validation   Green: test",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    return figure


def representative_replica(run: RunData) -> tuple[int, np.ndarray]:
    """Choose the test replica whose Transformer RMSE is nearest the median."""

    replica_rmse = np.sqrt(
        np.mean(np.square(run.transformer_predictions - run.targets), axis=1)
    )
    median_rmse = np.median(replica_rmse)
    replica = int(np.argmin(np.abs(replica_rmse - median_rmse)))
    return replica, replica_rmse


def plot_test_predictions(
    run: RunData,
    replica: int,
    replica_rmse: np.ndarray,
    segment_start: int,
    segment_length: int,
) -> plt.Figure:
    """Compare actual test values with Transformer and AR(1) forecasts."""

    test_length = run.targets.shape[1]
    if not 0 <= replica < run.values.shape[0]:
        raise ValueError(f"replica must be between 0 and {run.values.shape[0] - 1}.")
    if segment_start < 0 or segment_start >= test_length:
        raise ValueError(f"segment_start must be between 0 and {test_length - 1}.")
    segment_end = min(segment_start + segment_length, test_length)
    target_indices = np.arange(
        run.validation_end + segment_start, run.validation_end + segment_end
    )
    slice_ = slice(segment_start, segment_end)
    lagged = run.values[replica, target_indices - 1]

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(
        target_indices,
        run.targets[replica, slice_],
        color="#303030",
        linewidth=1.3,
        marker="o",
        markersize=2.5,
        label="Actual AR(1)",
        zorder=4,
    )
    axis.plot(
        target_indices,
        run.transformer_predictions[replica, slice_],
        color="#4C78A8",
        linewidth=2.1,
        label="Transformer",
        zorder=3,
    )
    axis.plot(
        target_indices,
        run.data_config.phi * lagged,
        color="#F58518",
        linewidth=1.7,
        linestyle="--",
        label=f"Oracle AR(1), $\\phi={run.data_config.phi:.3f}$",
    )
    axis.plot(
        target_indices,
        run.estimated_phi * lagged,
        color="#54A24B",
        linewidth=1.4,
        linestyle=":",
        label=f"Estimated AR(1), $\\hat{{\\phi}}={run.estimated_phi:.3f}$",
    )
    axis.set_title(
        f"One-step test forecasts — replica {replica + 1} "
        f"(Transformer RMSE={replica_rmse[replica]:.3f}, median-selected)"
    )
    axis.set_xlabel("Target time index")
    axis.set_ylabel("Value")
    axis.grid(alpha=0.2)
    axis.legend(ncol=2, frameon=False)
    figure.tight_layout()
    return figure


def plot_recovered_relationship(run: RunData) -> plt.Figure:
    """Plot Transformer forecasts against the last observation in each window."""

    lagged = run.values[:, run.validation_end - 1 : -1].reshape(-1)
    predictions = run.transformer_predictions.reshape(-1)
    rng = np.random.default_rng(0)
    sample_size = min(5_000, lagged.size)
    sample_indices = rng.choice(lagged.size, size=sample_size, replace=False)

    lower, upper = np.quantile(lagged, [0.005, 0.995])
    line_x = np.linspace(lower, upper, 200)
    bin_edges = np.quantile(lagged, np.linspace(0.0, 1.0, 26))
    bin_indices = np.digitize(lagged, np.unique(bin_edges)[1:-1])
    bin_x = []
    bin_y = []
    for bin_index in np.unique(bin_indices):
        members = bin_indices == bin_index
        bin_x.append(float(lagged[members].mean()))
        bin_y.append(float(predictions[members].mean()))

    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(
        lagged[sample_indices],
        predictions[sample_indices],
        s=7,
        alpha=0.10,
        color="#4C78A8",
        edgecolors="none",
        label="Transformer forecasts (sample)",
    )
    axis.plot(
        bin_x,
        bin_y,
        color="#1F4E79",
        marker="o",
        markersize=4,
        linewidth=2,
        label="Binned Transformer mean",
    )
    axis.plot(
        line_x,
        run.data_config.phi * line_x,
        color="#F58518",
        linewidth=2,
        linestyle="--",
        label=f"Theoretical $0.8x_t$",
    )
    axis.plot(
        line_x,
        run.estimated_phi * line_x,
        color="#54A24B",
        linewidth=1.5,
        linestyle=":",
        label=f"Estimated $\\hat{{\\phi}}x_t$ ({run.estimated_phi:.3f})",
    )
    axis.set_xlim(lower, upper)
    axis.set_ylim(run.data_config.phi * lower, run.data_config.phi * upper)
    axis.set_title("Relationship recovered by the Transformer on the test set")
    axis.set_xlabel("Last observed value $x_t$")
    axis.set_ylabel("One-step forecast $\\hat{x}_{t+1}$")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure


def create_figures(
    run_directory: Path,
    num_series: int = 4,
    replica: int | None = None,
    segment_start: int = 0,
    segment_length: int = 100,
    overwrite: bool = False,
) -> list[Path]:
    """Create the three requested visual checks and return their paths."""

    run = load_run(run_directory)
    selected_replica, replica_rmse = representative_replica(run)
    if replica is not None:
        selected_replica = replica
    figure_directory = run_directory / "figures"
    figure_directory.mkdir(exist_ok=True)
    outputs = [
        figure_directory / "generated_series.png",
        figure_directory / "test_predictions.png",
        figure_directory / "recovered_ar_relationship.png",
    ]
    figures = [
        plot_generated_series(run, num_series),
        plot_test_predictions(
            run,
            selected_replica,
            replica_rmse,
            segment_start,
            segment_length,
        ),
        plot_recovered_relationship(run),
    ]
    for figure, output in zip(figures, outputs, strict=True):
        save_figure(figure, output, overwrite)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--num-series", type=int, default=4)
    parser.add_argument(
        "--replica",
        type=int,
        help="Zero-based replica index; default selects the median-RMSE replica.",
    )
    parser.add_argument("--segment-start", type=int, default=0)
    parser.add_argument("--segment-length", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = create_figures(
        run_directory=args.run_directory,
        num_series=args.num_series,
        replica=args.replica,
        segment_start=args.segment_start,
        segment_length=args.segment_length,
        overwrite=args.overwrite,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
