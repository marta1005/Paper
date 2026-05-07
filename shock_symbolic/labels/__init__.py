"""Pseudo-label generation for symbolic sensor training."""

from shock_symbolic.labels.shock_labels import generate_shock_labels
from shock_symbolic.labels.separation_labels import generate_separation_labels

__all__ = ["generate_shock_labels", "generate_separation_labels"]
