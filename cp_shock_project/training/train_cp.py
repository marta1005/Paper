from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cp_shock_project.data.case_indexing import build_case_index
from cp_shock_project.data.dataset import CpSurfaceDataset, DatasetScalers
from cp_shock_project.data.load_arrays import load_arrays
from cp_shock_project.data.splits import point_indices_for_cases, train_val_case_split
from cp_shock_project.graph.graph_cache import load_graph
from cp_shock_project.graph.knn_graph import KNNGraphBuilder
from cp_shock_project.features.scaling import StandardScaler
from cp_shock_project.metrics.per_case import per_case_metrics
from cp_shock_project.metrics.regression import regression_metrics
from cp_shock_project.metrics.sensor_metrics import sensor_metrics
from cp_shock_project.metrics.shock_metrics import gradient_error, shock_region_metrics
from cp_shock_project.models import (
    BaselineCpMLP,
    FourierGraphCpNet,
    OracleGatedGraphFourierResidualCpNet,
    SymbolicGatedGraphFourierResidualCpNet,
    SymbolicWeightedGraphFourierCpNet,
)
from cp_shock_project.symbolic.symbolic_module import DummyShockSensor, SymbolicShockSensorModule
from cp_shock_project.training.checkpoints import save_checkpoint
from cp_shock_project.training.losses import cp_loss_components
from cp_shock_project.utils.config import save_config
from cp_shock_project.utils.io import ensure_dir
from cp_shock_project.utils.seed import seed_everything


def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    model_cfg = config.get("model", {})
    name = model_cfg.get("name", "baseline_mlp")
    kwargs = dict(model_cfg.get("kwargs", {}))
    if name == "baseline_mlp":
        return BaselineCpMLP(**kwargs)
    if name == "fourier_graph":
        return FourierGraphCpNet(**kwargs)
    if name == "symbolic_weighted_graph":
        return SymbolicWeightedGraphFourierCpNet(**kwargs)
    if name == "symbolic_gated_residual":
        sensor_cfg = config.get("symbolic_sensor", {})
        sensor_path = sensor_cfg.get("path")
        if sensor_path and Path(sensor_path).exists():
            sensor = SymbolicShockSensorModule.from_json(sensor_path)
        else:
            sensor = DummyShockSensor(float(sensor_cfg.get("dummy_value", 0.0)))
        return SymbolicGatedGraphFourierResidualCpNet(symbolic_sensor=sensor, **kwargs)
    if name == "oracle_gated_residual":
        return OracleGatedGraphFourierResidualCpNet(symbolic_sensor=DummyShockSensor(0.0), **kwargs)
    raise ValueError(f"Unknown model name: {name}")


def load_optional_vector(path: str | None, key: str | None = None) -> np.ndarray | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    if p.suffix == ".npy":
        return np.load(p, mmap_mode="r")
    data = np.load(p)
    if key is not None and key in data:
        return data[key]
    first = data.files[0]
    return data[first]


def prepare_datasets(config: dict[str, Any], split: str = "train") -> tuple[CpSurfaceDataset, CpSurfaceDataset | None, Any]:
    data_cfg = config.get("data", {})
    arrays = load_arrays(data_cfg.get("data_dir", "data"), mmap=True)
    X = arrays.X_train if split == "train" else arrays.X_test
    Y = arrays.Y_train if split == "train" else arrays.Y_test
    case_index = build_case_index(X)
    graph_cfg = config.get("graph", {})
    graph_path = graph_cfg.get("train_graph_path" if split == "train" else "test_graph_path")
    if graph_path and Path(graph_path).exists():
        graph = load_graph(graph_path)
    elif graph_cfg.get("build_if_missing", True):
        graph = KNNGraphBuilder(
            k_neighbors=int(graph_cfg.get("k_neighbors", 8)),
            chunk_size=graph_cfg.get("chunk_size"),
            projection=graph_cfg.get("projection", "xy"),
            coordinate_columns=graph_cfg.get("coordinate_columns"),
        ).build(
            X,
            case_index,
            max_cases=graph_cfg.get("max_cases"),
            max_points_per_case=graph_cfg.get("max_points_per_case"),
        )
    else:
        graph = None
    shock_cfg = config.get("shock_score", {})
    oracle = load_optional_vector(shock_cfg.get("train_path" if split == "train" else "test_path"), key="oracle_shock_score")
    symbolic_chi = load_optional_vector(config.get("symbolic_sensor", {}).get("chi_train_path" if split == "train" else "chi_test_path"), key="symbolic_chi")
    scaling_cfg = config.get("scaling", {})
    scalers = DatasetScalers()
    scaler_dir = Path(scaling_cfg.get("scaler_dir", "processed/scalers"))
    if split != "train":
        if scaling_cfg.get("enabled", False):
            scalers = DatasetScalers(
                x_scaler=StandardScaler.load(scaling_cfg.get("x_scaler_path", scaler_dir / "x_scaler.json")),
                cp_scaler=StandardScaler.load(scaling_cfg.get("cp_scaler_path", scaler_dir / "cp_scaler.json")),
            )
        idx = np.arange(X.shape[0])
        if data_cfg.get("max_eval_points"):
            idx = idx[: int(data_cfg["max_eval_points"])]
        ds = CpSurfaceDataset(
            X,
            Y,
            indices=idx,
            case_ids=case_index.case_ids,
            neighbor_indices=graph.neighbor_indices if graph else None,
            neighbor_distances=graph.neighbor_distances if graph else None,
            oracle_shock_score=oracle,
            symbolic_chi=symbolic_chi,
            scalers=scalers,
        )
        return ds, None, case_index
    train_cases, val_cases = train_val_case_split(case_index, data_cfg.get("val_fraction", 0.2), data_cfg.get("seed", 42))
    train_idx = point_indices_for_cases(case_index, train_cases)
    val_idx = point_indices_for_cases(case_index, val_cases)
    if data_cfg.get("max_train_points"):
        train_idx = train_idx[: int(data_cfg["max_train_points"])]
    if data_cfg.get("max_val_points"):
        val_idx = val_idx[: int(data_cfg["max_val_points"])]
    if scaling_cfg.get("enabled", False):
        scaler_dir.mkdir(parents=True, exist_ok=True)
        x_scaler = StandardScaler().fit(np.asarray(X[train_idx, :9], dtype=np.float32))
        cp_scaler = StandardScaler().fit(np.asarray(Y[train_idx, 0:1], dtype=np.float32))
        x_scaler.save(scaling_cfg.get("x_scaler_path", scaler_dir / "x_scaler.json"))
        cp_scaler.save(scaling_cfg.get("cp_scaler_path", scaler_dir / "cp_scaler.json"))
        scalers = DatasetScalers(x_scaler=x_scaler, cp_scaler=cp_scaler)
    common = dict(
        case_ids=case_index.case_ids,
        neighbor_indices=graph.neighbor_indices if graph else None,
        neighbor_distances=graph.neighbor_distances if graph else None,
        oracle_shock_score=oracle,
        symbolic_chi=symbolic_chi,
        scalers=scalers,
    )
    return CpSurfaceDataset(X, Y, indices=train_idx, **common), CpSurfaceDataset(X, Y, indices=val_idx, **common), case_index


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    lambdas: dict[str, float],
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    n = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        with torch.set_grad_enabled(training):
            outputs = model(**batch)
            losses = cp_loss_components(outputs, batch, lambdas=lambdas)
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                optimizer.step()
        bs = int(batch["X"].shape[0])
        n += bs
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * bs
    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    out: dict[str, list[np.ndarray]] = {"Cp_pred": [], "Cp": [], "case_id": [], "point_id": []}
    optional = ["oracle_shock_score", "symbolic_chi", "chi", "Cp_smooth", "delta_Cp"]
    for batch in loader:
        batch = batch_to_device(batch, device)
        pred = model(**batch)
        out["Cp_pred"].append(pred["Cp_pred"].detach().cpu().numpy().reshape(-1))
        out["Cp"].append(batch["Cp"].detach().cpu().numpy().reshape(-1))
        out["point_id"].append(batch["point_id"].detach().cpu().numpy().reshape(-1))
        if "case_id" in batch:
            out["case_id"].append(batch["case_id"].detach().cpu().numpy().reshape(-1))
        for key in optional:
            src = pred if key in pred else batch
            if key in src:
                out.setdefault(key, []).append(src[key].detach().cpu().numpy().reshape(-1))
    return {k: np.concatenate(v) for k, v in out.items() if v}


def train_from_config(config: dict[str, Any]) -> dict[str, float]:
    seed_everything(int(config.get("seed", 42)))
    output_dir = ensure_dir(config.get("output_dir", "outputs/run"))
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    metrics_dir = ensure_dir(output_dir / "metrics")
    save_config(config, output_dir / "config_resolved.yaml")
    train_ds, val_ds, _ = prepare_datasets(config, split="train")
    train_loader = DataLoader(train_ds, batch_size=int(config.get("training", {}).get("batch_size", 256)), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(config.get("training", {}).get("batch_size", 256)), shuffle=False, num_workers=0) if val_ds is not None else None
    device = device_auto()
    model = build_model(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config.get("training", {}).get("lr", 1e-3)), weight_decay=float(config.get("training", {}).get("weight_decay", 1e-6)))
    epochs = int(config.get("training", {}).get("epochs", 1))
    lambdas = config.get("loss", {})
    train_rows = []
    val_rows = []
    best_val = float("inf")
    best_metrics: dict[str, float] = {}
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, opt, device, lambdas)
        train_metrics["epoch"] = epoch
        train_rows.append(train_metrics)
        if val_loader is not None:
            val_metrics = run_epoch(model, val_loader, None, device, lambdas)
            val_metrics["epoch"] = epoch
            val_rows.append(val_metrics)
            score = val_metrics["total"]
        else:
            score = train_metrics["total"]
        save_checkpoint(ckpt_dir / "last.pt", model, opt, epoch, {"score": score}, config)
        if score < best_val:
            best_val = score
            best_metrics = {"best_epoch": epoch, "best_score": score}
            save_checkpoint(ckpt_dir / "best.pt", model, opt, epoch, best_metrics, config)
    pd.DataFrame(train_rows).to_csv(metrics_dir / "train.csv", index=False)
    if val_rows:
        pd.DataFrame(val_rows).to_csv(metrics_dir / "val.csv", index=False)
    return best_metrics


def evaluate_from_config(config: dict[str, Any]) -> tuple[dict[str, float], pd.DataFrame, dict[str, float]]:
    output_dir = ensure_dir(config.get("output_dir", "outputs/evaluation"))
    ds, _, case_index = prepare_datasets(config, split="test")
    loader = DataLoader(ds, batch_size=int(config.get("evaluation", {}).get("batch_size", 512)), shuffle=False, num_workers=0)
    device = device_auto()
    model = build_model(config).to(device)
    ckpt_path = config.get("checkpoint")
    if ckpt_path:
        from cp_shock_project.training.checkpoints import load_checkpoint

        load_checkpoint(ckpt_path, model, map_location=device)
    pred = predict(model, loader, device)
    if ds.scalers.cp_scaler is not None:
        pred["Cp"] = ds.scalers.cp_scaler.inverse_transform(pred["Cp"].reshape(-1, 1)).reshape(-1)
        pred["Cp_pred"] = ds.scalers.cp_scaler.inverse_transform(pred["Cp_pred"].reshape(-1, 1)).reshape(-1)
        if "Cp_smooth" in pred:
            pred["Cp_smooth"] = ds.scalers.cp_scaler.inverse_transform(pred["Cp_smooth"].reshape(-1, 1)).reshape(-1)
        if "delta_Cp" in pred and ds.scalers.cp_scaler.scale_ is not None:
            pred["delta_Cp"] = pred["delta_Cp"] * float(ds.scalers.cp_scaler.scale_[0])
    global_metrics = regression_metrics(pred["Cp"], pred["Cp_pred"], prefix="Cp")
    case_ids = pred.get("case_id", case_index.case_ids[pred["point_id"]])
    per_case_df, per_summary = per_case_metrics(pred["Cp"], pred["Cp_pred"], case_ids)
    global_metrics.update(per_summary)
    shock = {}
    if "oracle_shock_score" in pred:
        shock = shock_region_metrics(pred["Cp"], pred["Cp_pred"], pred["oracle_shock_score"], threshold=float(config.get("evaluation", {}).get("shock_threshold", 0.5)))
        if ds.neighbor_indices is not None and ds.neighbor_distances is not None:
            n_total = ds.X.shape[0]
            full_true = np.full(n_total, np.nan, dtype=np.float32)
            full_pred = np.full(n_total, np.nan, dtype=np.float32)
            full_oracle = np.zeros(n_total, dtype=np.float32)
            full_true[pred["point_id"]] = pred["Cp"]
            full_pred[pred["point_id"]] = pred["Cp_pred"]
            full_oracle[pred["point_id"]] = pred["oracle_shock_score"]
            shock["gradient_error_global"] = gradient_error(full_true, full_pred, ds.neighbor_indices, ds.neighbor_distances)
            shock["gradient_error_shock"] = gradient_error(
                full_true,
                full_pred,
                ds.neighbor_indices,
                ds.neighbor_distances,
                mask=full_oracle >= float(config.get("evaluation", {}).get("shock_threshold", 0.5)),
            )
    sensor = {}
    chi_key = "chi" if "chi" in pred else "symbolic_chi" if "symbolic_chi" in pred else None
    if chi_key is not None and "oracle_shock_score" in pred:
        sensor = sensor_metrics(pred[chi_key], pred["oracle_shock_score"], threshold=float(config.get("evaluation", {}).get("shock_threshold", 0.5)))
    ensure_dir(output_dir)
    from cp_shock_project.utils.io import save_json

    save_json(global_metrics, output_dir / "global_metrics.json")
    per_case_df.to_csv(output_dir / "per_case_metrics.csv", index=False)
    save_json(shock, output_dir / "shock_region_metrics.json")
    save_json(sensor, output_dir / "sensor_metrics.json")
    np.savez_compressed(output_dir / "predictions.npz", **pred)
    return global_metrics, per_case_df, shock
