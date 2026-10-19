from experiments.experiment_01_ar1.baselines import estimate_phi
from experiments.experiment_01_ar1.config import DataConfig
from experiments.experiment_01_ar1.data import generate_ar1, split_boundaries


def test_pooled_estimator_recovers_phi() -> None:
    config = DataConfig(num_replicas=100, series_length=500, seed=19)
    values = generate_ar1(config)
    train_end, _ = split_boundaries(config)

    estimated = estimate_phi(values, train_end)

    assert abs(estimated - config.phi) < 0.02
