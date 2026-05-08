# Shock-Aware Cp Prediction for ONERA CRM WBPN

Proyecto Python/PyTorch para predecir `Cp` sobre datos de superficie pointwise/scattered de ONERA CRM WBPN. El objetivo científico es reducir el suavizado de ondas de choque que aparece en MLPs pointwise estándar al combinar Fourier features, contexto local kNN y un sensor simbólico deployable.

## Datos

El pipeline espera:

```text
data/
  X_train.npy
  X_test.npy
  Ytrain.npy
  Ytest.npy
```

Columnas de `X`: `x, y, z, nx, ny, nz, Mach, AoA, pi`. Columnas de `Y`: `Cp, Cfx, Cfy, Cfz`. Los arrays se cargan con `np.load(..., mmap_mode="r")`.

El split nunca se hace por punto. Train/test vienen de los archivos originales, y la validación se crea separando condiciones CFD completas agrupadas por `(Mach, AoA, pi)`.

## Arquitectura

El modelo principal es `SymbolicGatedGraphFourierResidualCpNet`:

```text
chi = symbolic_shock_sensor(X)
phi = FourierFeatures(X)
h_graph = LocalGraphEncoder(X, phi, neighbors)
Cp_smooth = SmoothCpNetwork(X, phi, h_graph)
delta_Cp = ShockResidualNetwork(X, phi, h_graph, chi)
Cp_pred = Cp_smooth + chi * delta_Cp
```

También incluye:

- `BaselineCpMLP`: MLP pointwise.
- `FourierGraphCpNet`: Fourier + kNN/message passing sin sensor.
- `SymbolicWeightedGraphFourierCpNet`: usa `chi` para ponderar la loss.
- `OracleGatedGraphFourierResidualCpNet`: upper-bound usando el oracle score como gate.

## Oracle vs Sensor Deployable

El `oracle_shock_score` usa `Cp` real y solo se calcula en preparación, entrenamiento, validación y análisis:

```text
grad_Cp_mag_approx(i) = mean_j |Cp_j - Cp_i| / (||r_j - r_i|| + eps)
```

Se normaliza por percentiles dentro de cada condición CFD y se umbraliza para crear `shock_label`.

El sensor deployable se entrena con PySR:

```text
chi = g_symbolic(x, y, z, nx, ny, nz, Mach, AoA, pi)
```

No usa `Cp` real en inferencia. Si PySR no está instalado o Julia no está configurado, el wrapper falla con un error claro; el resto del framework puede seguir en modo oracle o dummy para tests.

## Grafo kNN

`KNNGraphBuilder` agrupa por `(Mach, AoA, pi)` y construye vecinos usando solo coordenadas 3D `x, y, z`. Nunca mezcla puntos de condiciones diferentes. Guarda `neighbor_indices` y `neighbor_distances` en `processed/graphs/`.

## Losses

El entrenamiento implementa MSE global, MAE, Huber, MSE ponderada por sensor, loss de región shock con oracle, loss non-shock, regularización del residual y una loss gradient-aware:

```text
L_grad = mean_i mean_j |
  ((Cp_pred_j - Cp_pred_i) - (Cp_true_j - Cp_true_i)) / (dist_ij + eps)
|
```

`L_grad` está implementada para tensores completos con grafo kNN; los YAML de ejemplo la dejan desactivada en el loop mini-batch básico para mantener el smoke training ligero.

## Orden de Ejecución

1. Inspeccionar arrays:

```bash
python scripts/00_inspect_arrays.py --config configs/data.yaml
```

2. Construir índice de casos:

```bash
python scripts/01_build_case_index.py --config configs/data.yaml
```

3. Construir grafos kNN:

```bash
python scripts/02_build_knn_graphs.py --config configs/graph.yaml
```

4. Calcular oracle shock score:

```bash
python scripts/03_compute_oracle_shock_score.py --config configs/shock_score.yaml
```

5. Construir dataset del sensor simbólico:

```bash
python scripts/04_build_symbolic_sensor_dataset.py --config configs/symbolic_sensor.yaml
```

6. Entrenar sensor simbólico:

```bash
python scripts/05_train_symbolic_sensor.py --config configs/symbolic_sensor.yaml
```

7. Entrenar baseline MLP:

```bash
python scripts/06_train_cp_model.py --config configs/baseline_mlp.yaml
```

8. Entrenar Fourier Graph model:

```bash
python scripts/06_train_cp_model.py --config configs/fourier_graph.yaml
```

9. Entrenar Symbolic Weighted Graph model:

```bash
python scripts/06_train_cp_model.py --config configs/symbolic_weighted_graph.yaml
```

10. Entrenar modelo principal:

```bash
python scripts/06_train_cp_model.py --config configs/symbolic_gated_residual.yaml
```

11. Entrenar upper-bound oracle:

```bash
python scripts/06_train_cp_model.py --config configs/oracle_gated_residual.yaml
```

12. Evaluar:

```bash
python scripts/07_evaluate_cp_model.py --config configs/evaluation.yaml
```

13. Generar plots:

```bash
python scripts/08_plot_predictions.py --config configs/evaluation.yaml
```

14. Exportar sensor simbólico:

```bash
python scripts/09_export_symbolic_sensor.py --config configs/symbolic_sensor.yaml
```

## Métricas

Se guardan métricas globales `MAE_Cp`, `RMSE_Cp`, `R2_Cp`, métricas por caso con `wrMAE_Cp`, métricas shock/non-shock y métricas del sensor (`correlation`, `MAE`, `F1`, `precision`, `recall`, `IoU`).

## Tests

Los tests usan datos sintéticos con varias condiciones, superficie fake y salto brusco de `Cp`:

```bash
pytest
```

Incluyen smoke training de 1 epoch para `BaselineCpMLP`, `FourierGraphCpNet` y `SymbolicGatedGraphFourierResidualCpNet`.
