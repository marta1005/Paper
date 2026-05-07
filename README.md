# Shock Symbolic Sensor

Pointwise/scattered pipeline for an interpretable symbolic shock-footprint sensor on ONERA CRM WBPN CFD data, following the pointwise array convention of the ONERA CRM WBPN database paper: <https://arxiv.org/abs/2505.06265>.

This repository now focuses on a symbolic sensor rather than a final neural network. It loads the native pointwise arrays:

```text
data/
  X_train.npy
  X_test.npy
  Y_train.npy or Ytrain.npy
  Y_test.npy or Ytest.npy
```

Column convention:

```text
X[:, 0:3] = x, y, z
X[:, 3:6] = nx, ny, nz
X[:, 6]   = Mach
X[:, 7]   = AoA
X[:, 8]   = pi_scaled

Y[:, 0] = Cp
Y[:, 1] = Cfx
Y[:, 2] = Cfy
Y[:, 3] = Cfz
```

The code accepts the current local aliases `Ytrain.npy` and `Ytest.npy`.

## Installation

Core smoke tests only require NumPy, PyYAML and Matplotlib:

```bash
pip install -e .
```

For the full symbolic pipeline:

```bash
pip install pandas scikit-learn scipy pyarrow pysr
```

`PySR` is imported only by `shock_symbolic/symbolic/pysr_trainer.py`. If it is missing, the training script fails with a clear message; the data, feature, label and table stages still work.

## Command Order

1. Inspect arrays:

```bash
python scripts/00_inspect_arrays.py --config configs/data.yaml
```

2. Build CFD condition index:

```bash
python scripts/01_build_case_index.py --config configs/data.yaml
```

3. Compute pointwise features:

```bash
python scripts/02_compute_features.py --config configs/features.yaml
```

4. Generate pseudo-labels:

```bash
python scripts/03_generate_labels.py --config configs/labels.yaml
```

5. Build symbolic regression dataset:

```bash
python scripts/04_build_symbolic_dataset.py --config configs/symbolic_dataset.yaml
```

6. Train symbolic sensor with PySR:

```bash
python scripts/05_train_symbolic_sensor.py --config configs/pysr.yaml
```

7. Evaluate symbolic sensor:

```bash
python scripts/06_evaluate_symbolic_sensor.py --config configs/pysr.yaml
```

8. Export formula:

```bash
python scripts/07_export_sensor.py --config configs/pysr.yaml
```

9. Plot a grid of critical Cp cases:

```bash
python scripts/08_plot_critical_cp_grid.py --config configs/critical_cp_grid.yaml
```

By default this selects transonic cases (`Mach >= 0.7`), sorts by high Mach and high `|AoA|`, and plots the upper surface in 2D planform. If you provide `prediction.path` in `configs/critical_cp_grid.yaml`, the grid becomes:

```text
row = critical CFD case
columns = true Cp | predicted Cp | predicted - true
```

If `prediction.path` is empty, it produces a truth-only critical Cp grid so you can verify that the wing and critical cases are displayed correctly.

Accepted prediction files:

```text
1. .npy or .npz full split vector:
   cp_pred.shape == (N_split_points,)

2. .npy or .npz full split output:
   y_pred.shape == (N_split_points, 4)
   column: 0 selects Cp

3. .npz with per-case arrays:
   test_0012 = cp_pred_for_case_0012
   test_0013 = cp_pred_for_case_0013
```

For future model outputs, set:

```yaml
prediction:
  path: outputs/path/to/your_cp_prediction.npy
  key:
  column: 0
```

## Main Outputs

```text
outputs/symbolic/inspect/arrays.json
outputs/symbolic/case_index/case_index_train.csv
outputs/symbolic/features/{train,test}/{case_id}.npz
outputs/symbolic/labels/{train,test}/{case_id}.npz
outputs/symbolic/tables/shock_symbolic_train.csv or .parquet
outputs/symbolic/pysr/equations.csv
outputs/symbolic/pysr/best_equation.txt
outputs/symbolic/pysr/best_equation.tex
outputs/symbolic/pysr/best_sensor.json
outputs/symbolic/pysr/threshold.json
outputs/symbolic/evaluation/test/per_case_metrics.csv
outputs/symbolic/evaluation/test/global_metrics.json
```

## Features

Features are computed within each CFD condition only:

- `Cp`
- `Cfx`, `Cfy`, `Cfz`
- `Cf_mag`
- `Cf_parallel`, using freestream direction projected onto the local tangent plane
- `Cf_perp`
- `Cf_angle_stream`
- `grad_Cp_mag`, kNN weighted local finite difference in 3D
- `grad_Cp_streamwise`
- `local_Cp_contrast`
- `grad_Cf_mag`
- `local_Cf_contrast`
- `x`, `y`, `z`
- `nx`, `ny`, `nz`
- `Mach`, `AoA`, `pi_scaled`

If `scikit-learn` is installed, kNN uses `NearestNeighbors`. Without it, a NumPy fallback is used only for small sampled snapshots.

## Labels

Shock pseudo-labels are generated per condition:

- compute `grad_Cp_mag`,
- threshold by percentile, e.g. `98.5`,
- optionally suppress labels for `Mach <= 0.7`,
- compute continuous `shock_score` by robust percentile normalization.

Separation labels use:

- low `Cf_mag`,
- high `grad_Cf_mag`,
- reversed `Cf_parallel`.

## Huge Data Notes

The original arrays are opened with `mmap_mode="r"`. The heavy stage is kNN, so tune:

```yaml
features:
  max_cases: 2
  max_points_per_case: 3000
  k_neighbors: 16
  batch_size: 4096
```

For full production runs, install `scikit-learn`, increase `max_points_per_case`, and set `max_cases: null`.

The default feature config starts with `case_filters.min_mach: 0.7` so quick runs do not spend time on clearly subsonic cases when generating shock sensors.

## Tests

```bash
pytest
```

Tests use small synthetic point clouds and do not require PySR, pandas or scikit-learn.
