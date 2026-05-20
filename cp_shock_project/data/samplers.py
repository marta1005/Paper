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


class RandomSubsetSampler(Sampler[int]):
    """Draw a fresh random subset of dataset positions each epoch.

    This avoids constructing a shuffled permutation of tens of millions of
    points while still sampling from the full training pool over time.
    """

    def __init__(self, data_source_len: int, num_samples: int, seed: int = 42, replacement: bool = False, chunk_size: int = 100_000):
        if data_source_len <= 0:
            raise ValueError("data_source_len must be positive")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.data_source_len = int(data_source_len)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.replacement = bool(replacement)
        self.chunk_size = int(chunk_size)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        if not self.replacement and self.num_samples <= self.data_source_len:
            # The subset array is modest compared with a full 81M-point permutation.
            idx = rng.choice(self.data_source_len, size=self.num_samples, replace=False)
            yield from idx.tolist()
            return
        remaining = self.num_samples
        while remaining > 0:
            n = min(self.chunk_size, remaining)
            yield from rng.integers(0, self.data_source_len, size=n).tolist()
            remaining -= n

    def __len__(self) -> int:
        return self.num_samples
