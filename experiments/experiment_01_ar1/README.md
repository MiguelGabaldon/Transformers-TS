# Experiment 01 — Recovering a stationary AR(1)

## Objective

This controlled experiment checks that a small PyTorch Transformer can learn
one-step forecasting for the known process

\[
x_t = 0.8x_{t-1} + \varepsilon_t, \qquad
\varepsilon_t \sim \mathcal N(0,1).
\]

This is an **AR(1) with coefficient \(\phi=0.8\)**; AR(1) denotes the model
order. The goal is implementation validation, not demonstrating that a
Transformer is preferable to the correctly specified linear model.

## Data and splitting

The default run generates 50 independent replicas of 2,000 observations. To
avoid a transient burn-in, every initial value is sampled from the stationary
distribution

\[
x_0 \sim \mathcal N\left(0, \frac{1}{1-0.8^2}\right).
\]

Each replica is split chronologically: 70% train (indices 0–1399), 15%
validation (1400–1699), and 15% test (1700–1999). The Transformer receives the
last \(L=24\) observations and predicts the next one. Validation and test
windows may use history immediately before their boundary, because it would be
known at forecast time, but targets never cross split boundaries. Normalization
is fitted only to raw training observations and shared across all replicas.

One model is trained jointly on windows from all replicas. Validation controls
early stopping; the test split is used only for final reporting.

## Model and references

[`model.py`](model.py) contains an encoder-only Transformer. A scalar input
projection maps each observation to `d_model=32`; learned positional embeddings
are followed by two Transformer encoder layers with four attention heads. A
linear head maps the final observed token to the one-step forecast. No causal
mask is required: the entire input window is observed and contains no target or
future values.

Two analytic references use the same forecast targets:

- **Oracle AR(1):** predicts \(0.8x_t\).
- **Estimated AR(1):** estimates one shared zero-intercept coefficient by
  pooled least squares using training observations only.

Both references are useful: the estimated model is realistic, while the oracle
is the theoretical check. Results report RMSE and MAE over all forecasts plus
the mean and sample standard deviation of those metrics across the 50 replicas.

## Running

From the repository root:

```powershell
python -m experiments.experiment_01_ar1.train
```

Key data settings and the epoch limit are CLI-configurable; run `--help` for
options. Every execution creates a timestamped directory under `results/` with
the generated series in `data.npz`, the exact `config.json`, structured
`metrics.json`, and the `model.pt` checkpoint. Earlier experimental results are
not overwritten. Defaults are explicit in [`config.py`](config.py).

Run the focused tests with:

```powershell
python -m pytest experiments/experiment_01_ar1/tests
```

Create the generated-series, test-prediction, and recovered-relationship visual
checks for a completed run with:

```powershell
python -m experiments.experiment_01_ar1.plot_results path/to/run_directory
```

Figures are saved inside that run's `figures/` directory. The prediction plot
uses the median-RMSE test replica by default to avoid cherry-picking.
