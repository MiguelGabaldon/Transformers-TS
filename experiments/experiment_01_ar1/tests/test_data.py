import numpy as np

from experiments.experiment_01_ar1.config import DataConfig
from experiments.experiment_01_ar1.data import (
    fit_training_standardizer,
    generate_ar1,
    make_windowed_split,
    split_boundaries,
)


def test_generation_is_reproducible_and_approximately_stationary() -> None:
    config = DataConfig(num_replicas=400, series_length=200, seed=7)
    first = generate_ar1(config)
    second = generate_ar1(config)

    np.testing.assert_array_equal(first, second)
    expected_variance = config.sigma**2 / (1.0 - config.phi**2)
    assert abs(float(first.var()) - expected_variance) < 0.2


def test_chronological_windows_are_aligned_without_target_leakage() -> None:
    config = DataConfig(
        num_replicas=1,
        series_length=20,
        lookback=4,
        train_fraction=0.5,
        validation_fraction=0.25,
    )
    values = np.arange(20, dtype=np.float32).reshape(1, -1)
    train_end, validation_end = split_boundaries(config)
    standardizer = fit_training_standardizer(values, train_end)

    train = make_windowed_split(values, config, "train", standardizer)
    validation = make_windowed_split(values, config, "validation", standardizer)
    test = make_windowed_split(values, config, "test", standardizer)

    assert train.inputs.shape == (train_end - config.lookback, config.lookback)
    assert validation.inputs.shape == (validation_end - train_end, config.lookback)
    assert test.inputs.shape == (config.series_length - validation_end, config.lookback)
    np.testing.assert_allclose(
        standardizer.inverse_transform(validation.inputs[0].numpy()),
        np.array([6, 7, 8, 9]),
    )
    assert standardizer.inverse_transform(validation.targets[0].numpy()) == 10
    assert standardizer.mean == 4.5


def test_each_split_has_the_expected_default_target_count() -> None:
    config = DataConfig()
    values = generate_ar1(config)
    train_end, _ = split_boundaries(config)
    standardizer = fit_training_standardizer(values, train_end)

    train = make_windowed_split(values, config, "train", standardizer)
    validation = make_windowed_split(values, config, "validation", standardizer)
    test = make_windowed_split(values, config, "test", standardizer)

    assert train.targets_per_replica == 1_400 - 24
    assert validation.targets_per_replica == 300
    assert test.targets_per_replica == 300
