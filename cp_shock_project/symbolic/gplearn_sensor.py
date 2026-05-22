from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any
import re

import numpy as np
import pandas as pd

from cp_shock_project.symbolic.expression import FEATURE_COLUMNS, SYMBOLIC_VARIABLES, SymbolicExpression
from cp_shock_project.utils.io import save_json

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cp_shock_matplotlib"))


def _load_sensor_tables(train_parquet: str | Path, val_parquet: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = pd.read_parquet(train_parquet).rename(columns={"pi": "pi_param"})
    val = pd.read_parquet(val_parquet).rename(columns={"pi": "pi_param"})
    missing = [c for c in SYMBOLIC_VARIABLES if c not in train.columns or c not in val.columns]
    if missing:
        raise ValueError(f"Missing symbolic sensor columns: {missing}")
    X_train = train[SYMBOLIC_VARIABLES].to_numpy(dtype=np.float64)
    y_train = train["oracle_shock_score"].to_numpy(dtype=np.float64)
    X_val = val[SYMBOLIC_VARIABLES].to_numpy(dtype=np.float64)
    y_val = val["oracle_shock_score"].to_numpy(dtype=np.float64)
    return train, val, X_train, y_train, X_val, y_val


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict[str, float]:
    true = np.asarray(y_true).reshape(-1) >= threshold
    pred = np.asarray(y_pred).reshape(-1) >= threshold
    tp = float(np.sum(true & pred))
    fp = float(np.sum(~true & pred))
    fn = float(np.sum(true & ~pred))
    tn = float(np.sum(~true & ~pred))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    iou = tp / (tp + fp + fn + 1e-12)
    out = {
        "precision": float(precision),
        "recall": float(recall),
        "F1": float(f1),
        "IoU": float(iou),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_shock_fraction": float(np.mean(pred)),
        "oracle_shock_fraction": float(np.mean(true)),
    }
    try:
        from sklearn.metrics import roc_auc_score

        if len(np.unique(true.astype(int))) > 1:
            out["roc_auc"] = float(roc_auc_score(true.astype(int), y_pred))
    except Exception:
        pass
    return out


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    err = np.abs(y_pred - y_true)
    shock = y_true >= threshold
    nonshock = ~shock
    out = {
        "mae_global": float(np.mean(err)),
        "rmse_global": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "prediction_min": float(np.min(y_pred)),
        "prediction_max": float(np.max(y_pred)),
        "prediction_mean": float(np.mean(y_pred)),
        "prediction_std": float(np.std(y_pred)),
        "target_min": float(np.min(y_true)),
        "target_max": float(np.max(y_true)),
        "target_mean": float(np.mean(y_true)),
        "target_std": float(np.std(y_true)),
        "correlation": float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_true) > 0 and np.std(y_pred) > 0 else 0.0,
        "n_shock": int(np.sum(shock)),
        "n_nonshock": int(np.sum(nonshock)),
    }
    out["mae_shock"] = float(np.mean(err[shock])) if np.any(shock) else float("nan")
    out["mae_nonshock"] = float(np.mean(err[nonshock])) if np.any(nonshock) else float("nan")
    out.update(_binary_metrics(y_true, y_pred, threshold))
    return out


def _has_self_subtraction(expression: str) -> bool:
    pattern = re.compile(r"sub\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*\1\s*\)")
    return bool(pattern.search(expression))


def _degeneracy_report(
    expression: str,
    y_pred: np.ndarray,
    metrics: dict[str, float],
    std_tolerance: float,
    range_tolerance: float,
    min_recall: float,
) -> dict[str, Any]:
    pred_range = float(np.max(y_pred) - np.min(y_pred))
    reasons: list[str] = []
    if float(np.std(y_pred)) < std_tolerance:
        reasons.append("prediction_std_below_tolerance")
    if pred_range < range_tolerance:
        reasons.append("prediction_range_below_tolerance")
    if _has_self_subtraction(expression):
        reasons.append("expression_contains_self_subtraction")
    if metrics.get("recall", 0.0) < min_recall:
        reasons.append("shock_recall_below_minimum")
    return {"is_degenerate": bool(reasons), "reasons": reasons}


def _write_validation_artifacts(root: Path, val: pd.DataFrame, y_val: np.ndarray, pred: np.ndarray, threshold: float, max_rows: int = 200_000) -> None:
    out = val[SYMBOLIC_VARIABLES].copy()
    out["oracle_shock_score"] = y_val
    out["symbolic_sensor_prediction"] = pred
    out["abs_error"] = np.abs(pred - y_val)
    out["oracle_shock_label"] = (y_val >= threshold).astype(np.int8)
    out["predicted_shock_label"] = (pred >= threshold).astype(np.int8)
    if len(out) > max_rows:
        out = out.sample(max_rows, random_state=42).reset_index(drop=True)
    out.to_csv(root / "validation_predictions.csv", index=False)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        diag_dir = root / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 5), dpi=160)
        ax.scatter(y_val, pred, s=2, alpha=0.25)
        ax.plot([0, 1], [0, 1], color="black", linewidth=1)
        ax.set_xlabel("oracle_shock_score")
        ax.set_ylabel("symbolic prediction")
        ax.set_title("Sensor validation")
        fig.tight_layout()
        fig.savefig(diag_dir / "oracle_vs_prediction.png")
        plt.close(fig)
        for field, title in [
            ("oracle_shock_score", "Oracle shock score"),
            ("symbolic_sensor_prediction", "Symbolic prediction"),
            ("abs_error", "Absolute error"),
            ("oracle_shock_label", "Oracle shock mask"),
            ("predicted_shock_label", "Predicted shock mask"),
        ]:
            fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
            sc = ax.scatter(out["x"], out["y"], c=out[field], s=2, cmap="coolwarm", linewidths=0)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(title)
            fig.colorbar(sc, ax=ax)
            fig.tight_layout()
            fig.savefig(diag_dir / f"{field}_xy.png")
            plt.close(fig)
    except Exception:
        pass


def train_gplearn_sensor(
    train_parquet: str | Path,
    val_parquet: str | Path,
    out_dir: str | Path,
    gplearn_kwargs: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Train a deployable symbolic shock sensor with gplearn."""
    try:
        from gplearn.genetic import SymbolicRegressor
    except Exception as exc:
        raise RuntimeError(
            "gplearn is not installed. Install it with `python -m pip install gplearn`, "
            "then rerun scripts/05_train_symbolic_sensor.py."
        ) from exc

    train, val, X_train, y_train, X_val, y_val = _load_sensor_tables(train_parquet, val_parquet)
    kwargs: dict[str, Any] = {
        "population_size": 1000,
        "generations": 30,
        "tournament_size": 20,
        "stopping_criteria": 0.0,
        "const_range": (-1.0, 1.0),
        "init_depth": (2, 5),
        "init_method": "half and half",
        "function_set": ("add", "sub", "mul", "div", "abs", "sqrt", "max", "min"),
        "metric": "mean absolute error",
        "parsimony_coefficient": 0.0005,
        "p_crossover": 0.85,
        "p_subtree_mutation": 0.05,
        "p_hoist_mutation": 0.03,
        "p_point_mutation": 0.05,
        "max_samples": 0.9,
        "feature_names": SYMBOLIC_VARIABLES,
        "n_jobs": 1,
        "verbose": 1,
        "random_state": 42,
    }
    kwargs.update(gplearn_kwargs or {})
    if isinstance(kwargs.get("function_set"), list):
        kwargs["function_set"] = tuple(kwargs["function_set"])
    if isinstance(kwargs.get("init_depth"), list):
        kwargs["init_depth"] = tuple(kwargs["init_depth"])
    if isinstance(kwargs.get("const_range"), list):
        kwargs["const_range"] = tuple(kwargs["const_range"])

    sensor_cfg = dict(gplearn_kwargs or {})
    threshold = float(sensor_cfg.get("shock_threshold", sensor_cfg.get("evaluation_threshold", 0.3)))
    sample_weight_alpha = float(sensor_cfg.get("sample_weight_alpha", 10.0))
    reject_degenerate = bool(sensor_cfg.get("reject_degenerate", True))
    std_tolerance = float(sensor_cfg.get("degenerate_std_tolerance", 1e-4))
    range_tolerance = float(sensor_cfg.get("degenerate_range_tolerance", 1e-3))
    min_recall = float(sensor_cfg.get("degenerate_min_recall", 0.05))
    for local_key in [
        "sample_weight_alpha",
        "reject_degenerate",
        "degenerate_std_tolerance",
        "degenerate_range_tolerance",
        "degenerate_min_recall",
        "shock_threshold",
        "evaluation_threshold",
    ]:
        kwargs.pop(local_key, None)
    sample_weight = 1.0 + sample_weight_alpha * y_train

    model = SymbolicRegressor(**kwargs)
    model.fit(X_train, y_train, sample_weight=sample_weight)

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    expression = str(model._program)
    pred = np.clip(model.predict(X_val), 0.0, 1.0)
    metrics = _regression_metrics(y_val, pred, threshold=threshold)
    degeneracy = _degeneracy_report(
        expression,
        pred,
        metrics,
        std_tolerance=std_tolerance,
        range_tolerance=range_tolerance,
        min_recall=min_recall,
    )
    program = model._program
    hall = pd.DataFrame(
        [
            {
                "backend": "gplearn",
                "expression": expression,
                "fitness": float(getattr(program, "fitness_", np.nan)),
                "raw_fitness": float(getattr(program, "raw_fitness_", np.nan)),
                "depth": int(program.depth_),
                "length": int(program.length_),
                "val_mae": metrics["mae_global"],
                "val_mae_shock": metrics["mae_shock"],
                "val_mae_nonshock": metrics["mae_nonshock"],
                "val_rmse": metrics["rmse_global"],
                "correlation": metrics["correlation"],
                "recall": metrics["recall"],
                "is_degenerate": degeneracy["is_degenerate"],
            }
        ]
    )
    hall.to_csv(root / "equations.csv", index=False)
    hall.to_csv(root / "hall_of_fame.csv", index=False)
    save_json(metrics, root / "sensor_metrics.json")
    save_json(degeneracy, root / "degeneracy_report.json")
    _write_validation_artifacts(root, val, y_val, pred, threshold=threshold)
    (root / "best_equation.txt").write_text(expression + "\n", encoding="utf-8")
    (root / "best_equation.tex").write_text(expression + "\n", encoding="utf-8")
    save_json(
        {
            "backend": "gplearn",
            "expression": expression,
            "variables": SYMBOLIC_VARIABLES,
            "feature_columns": FEATURE_COLUMNS,
            "clip_min": 0.0,
            "clip_max": 1.0,
            "val_mae": metrics["mae_global"],
            "val_rmse": metrics["rmse_global"],
            "metrics": metrics,
            "degeneracy": degeneracy,
        },
        root / "best_sensor.json",
    )
    save_json({"threshold": threshold, "target": "oracle_shock_score"}, root / "threshold.json")
    if degeneracy["is_degenerate"] and reject_degenerate:
        reasons = ", ".join(degeneracy["reasons"])
        raise RuntimeError(f"Rejected degenerate symbolic sensor: {reasons}. See {root / 'degeneracy_report.json'}")
    return {
        "equations": root / "equations.csv",
        "hall_of_fame": root / "hall_of_fame.csv",
        "best_equation": root / "best_equation.txt",
        "best_sensor": root / "best_sensor.json",
        "threshold": root / "threshold.json",
    }


def evaluate_gplearn_sensor(sensor_json: str | Path, X: np.ndarray) -> np.ndarray:
    return SymbolicExpression.from_json(sensor_json).evaluate(X)
