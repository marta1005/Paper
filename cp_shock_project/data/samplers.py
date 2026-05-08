from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler


class BalancedShockSampler(Sampler[int]):
    """Simple replacement sampler balancing shock and non-shock points."""

    def __init__(self, shock_score: np.ndarray, threshold: float = 0.5, num_samples: int | None = None, seed: int = 42):
        self.shock_idx = np.flatnonzero(shock_score > threshold)
        self.nonshock_idx = np.flatnonzero(shock_score <= threshold)
        self.num_samples = int(num_samples or len(shock_score))
        self.seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        half = self.num_samples // 2
        if len(self.shock_idx) == 0 or len(self.nonshock_idx) == 0:
            yield from rng.integers(0, self.num_samples, size=self.num_samples).tolist()
            return
        shock = rng.choice(self.shock_idx, size=half, replace=len(self.shock_idx) < half)
        nonshock = rng.choice(self.nonshock_idx, size=self.num_samples - half, replace=len(self.nonshock_idx) < self.num_samples - half)
        all_idx = np.concatenate([shock, nonshock])
        rng.shuffle(all_idx)
        yield from all_idx.tolist()

    def __len__(self) -> int:
        return self.num_samples
