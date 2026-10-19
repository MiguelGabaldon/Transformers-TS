"""Train and evaluate the Transformer and AR(1) baselines."""

import argparse
import copy
import json
import random
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .baselines import evaluate_ar1_baselines
from .config import DataConfig, ExperimentConfig, TrainingConfig
from .data import Standardizer, WindowedSplit, prepare_data, split_boundaries
from .metrics import summarize_replica_metrics
from .model import ARTransformer


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch for reproducible runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return device


def make_loader(
    split: WindowedSplit,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        split.dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def mean_loss(
    model: nn.Module, loader: DataLoader, loss_function: nn.Module, device: torch.device
) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            loss = loss_function(model(inputs), targets)
            total_loss += float(loss) * targets.numel()
            total_examples += targets.numel()
    return total_loss / total_examples


def fit_model(
    model: ARTransformer,
    splits: dict[str, WindowedSplit],
    config: TrainingConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    """Fit with validation early stopping and restore the best parameters."""

    train_loader = make_loader(
        splits["train"], config.batch_size, shuffle=True, seed=config.seed
    )
    validation_loader = make_loader(
        splits["validation"], config.batch_size, shuffle=False, seed=config.seed
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_function = nn.MSELoss()
    model.to(device)

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_examples = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss.detach()) * targets.numel()
            total_examples += targets.numel()

        validation_loss = mean_loss(
            model, validation_loader, loss_function, device
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_mse_standardized": total_train_loss / total_examples,
                "validation_mse_standardized": validation_loss,
            }
        )
        print(
            f"epoch={epoch:02d} "
            f"train_mse={history[-1]['train_mse_standardized']:.6f} "
            f"validation_mse={validation_loss:.6f}"
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("Training completed without producing a checkpoint.")
    model.load_state_dict(best_state)
    return history


def predict(
    model: ARTransformer,
    split: WindowedSplit,
    standardizer: Standardizer,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = make_loader(split, batch_size, shuffle=False, seed=0)
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for inputs, _ in loader:
            predictions.append(model(inputs.to(device)).cpu().numpy())
    predicted_standardized = np.concatenate(predictions)
    target_standardized = split.targets.numpy()
    shape = (split.num_replicas, split.targets_per_replica)
    return (
        standardizer.inverse_transform(target_standardized).reshape(shape),
        standardizer.inverse_transform(predicted_standardized).reshape(shape),
    )


def create_run_directory(root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_directory = root / timestamp
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def run_experiment(config: ExperimentConfig, output_root: Path) -> Path:
    """Execute one reproducible run and save its configuration and results."""

    seed_everything(config.training.seed)
    device = resolve_device(config.training.device)
    values, standardizer, splits = prepare_data(config.data)
    estimated_phi, baseline_metrics = evaluate_ar1_baselines(values, config.data)

    model = ARTransformer(config.data.lookback, config.model)
    history = fit_model(model, splits, config.training, device)
    transformer_metrics: dict[str, Any] = {}
    for split_name, split in splits.items():
        targets, predictions = predict(
            model,
            split,
            standardizer,
            config.training.batch_size,
            device,
        )
        transformer_metrics[split_name] = summarize_replica_metrics(
            targets, predictions
        )

    run_directory = create_run_directory(output_root)
    results = {
        "device": str(device),
        "estimated_phi": estimated_phi,
        "best_epoch": int(
            min(history, key=lambda item: item["validation_mse_standardized"])[
                "epoch"
            ]
        ),
        "metrics": {"transformer": transformer_metrics, **baseline_metrics},
        "training_history": history,
    }
    (run_directory / "config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    (run_directory / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    train_end, validation_end = split_boundaries(config.data)
    np.savez_compressed(
        run_directory / "data.npz",
        values=values,
        train_end=np.int64(train_end),
        validation_end=np.int64(validation_end),
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "lookback": config.data.lookback,
            "model_config": asdict(config.model),
            "standardizer": asdict(standardizer),
        },
        run_directory / "model.pt",
    )
    return run_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phi", type=float, default=0.8)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--num-replicas", type=int, default=50)
    parser.add_argument("--series-length", type=int, default=2_000)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        data=DataConfig(
            phi=args.phi,
            sigma=args.sigma,
            lookback=args.lookback,
            num_replicas=args.num_replicas,
            series_length=args.series_length,
        ),
        training=replace(
            TrainingConfig(), max_epochs=args.max_epochs, device=args.device
        ),
    )
    run_directory = run_experiment(config, args.output_root)
    print(f"Saved run to {run_directory}")


if __name__ == "__main__":
    main()
