import pytest
import torch

from experiments.experiment_01_ar1.config import ModelConfig
from experiments.experiment_01_ar1.metrics import summarize_replica_metrics
from experiments.experiment_01_ar1.model import ARTransformer


@pytest.mark.parametrize("batch_size", [1, 7])
def test_forward_and_backward_pass(batch_size: int) -> None:
    torch.manual_seed(2)
    model = ARTransformer(
        lookback=24,
        config=ModelConfig(
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        ),
    )
    inputs = torch.randn(batch_size, 24)
    targets = torch.randn(batch_size)

    predictions = model(inputs)
    loss = torch.nn.functional.mse_loss(predictions, targets)
    loss.backward()

    assert predictions.shape == (batch_size,)
    assert torch.isfinite(predictions).all()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_rejects_wrong_input_shape() -> None:
    model = ARTransformer(lookback=24, config=ModelConfig())
    with pytest.raises(ValueError, match="Expected inputs"):
        model(torch.randn(3, 23))


def test_single_replica_metric_summary_is_finite() -> None:
    targets = torch.tensor([[1.0, 2.0]]).numpy()
    predictions = torch.tensor([[1.5, 1.5]]).numpy()

    summary = summarize_replica_metrics(targets, predictions)

    assert summary["across_replicas"]["rmse_std"] == 0.0
    assert summary["across_replicas"]["mae_std"] == 0.0
