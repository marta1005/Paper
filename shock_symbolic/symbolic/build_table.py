"""Build balanced tabular datasets for symbolic regression."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from shock_symbolic.utils.io import save_table


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _sample_indices(
    label: np.ndarray,
    max_positive: int | None,
    negative_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    positive = np.flatnonzero(label > 0.5)
    negative = np.flatnonzero(label <= 0.5)
    if max_positive is not None and positive.size > max_positive:
        positive = rng.choice(positive, size=int(max_positive), replace=False)
    n_negative = int(max(1, round(max(positive.size, 1) * negative_ratio)))
    n_negative = min(n_negative, negative.size)
    negative = rng.choice(negative, size=n_negative, replace=False) if n_negative > 0 else np.empty(0, dtype=np.int64)
    return np.sort(np.concatenate([positive, negative]).astype(np.int64))


def build_symbolic_table(
    feature_files: list[str | Path],
    label_files: list[str | Path],
    output_base: str | Path,
    feature_names: list[str],
    target_name: str = "shock_score",
    max_positive_per_case: int | None = None,
    negative_ratio: float = 3.0,
    seed: int = 42,
) -> Path:
    """Build and save a balanced symbolic-regression table."""
    if len(feature_files) != len(label_files):
        raise ValueError("feature_files and label_files must have the same length")
    rng = np.random.default_rng(seed)
    columns: dict[str, list[np.ndarray]] = {name: [] for name in feature_names}
    columns[target_name] = []
    columns["shock_label"] = []
    columns["case_id"] = []
    columns["point_id"] = []
    columns["Mach"] = []
    columns["AoA"] = []
    columns["pi_scaled"] = []

    for feature_path, label_path in zip(feature_files, label_files):
        features = _load_npz(feature_path)
        labels = _load_npz(label_path)
        shock_label = np.asarray(labels["shock_label"], dtype=np.float32)
        idx = _sample_indices(shock_label, max_positive=max_positive_per_case, negative_ratio=negative_ratio, rng=rng)
        case_id = str(np.asarray(features.get("case_id", Path(feature_path).stem)))
        for name in feature_names:
            if name not in features:
                raise KeyError(f"Feature {name!r} not found in {feature_path}")
            columns[name].append(np.asarray(features[name])[idx])
        columns[target_name].append(np.asarray(labels[target_name], dtype=np.float32)[idx])
        columns["shock_label"].append(shock_label[idx])
        columns["case_id"].append(np.asarray([case_id] * idx.size))
        columns["point_id"].append(np.asarray(features["point_id"])[idx])
        for meta in ("Mach", "AoA", "pi_scaled"):
            columns[meta].append(np.asarray(features[meta], dtype=np.float32)[idx])

    table = {key: np.concatenate(parts) if parts else np.asarray([]) for key, parts in columns.items()}
    return save_table(output_base, table)
