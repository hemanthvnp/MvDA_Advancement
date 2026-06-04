"""Reproducibility and preprocessing helpers."""

from __future__ import annotations

import os
import random
from typing import Iterable, List

import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler


def set_seed(seed: int = 0) -> None:
    """Seed Python and NumPy RNGs for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_scaler(name: str):
    """Return a fresh, unfitted scaler by name.

    Parameters
    ----------
    name : {"robust", "standard", "none"}
    """
    name = (name or "none").lower()
    if name == "robust":
        return RobustScaler()
    if name == "standard":
        return StandardScaler()
    if name == "none":
        return None
    raise ValueError(f"Unknown scaler: {name!r}")


def fit_scalers(views: List[np.ndarray], name: str):
    """Fit one scaler per view on training data; return (scaled_views, scalers)."""
    scalers = []
    scaled = []
    for v in views:
        sc = get_scaler(name)
        if sc is None:
            scaled.append(v)
            scalers.append(None)
        else:
            scaled.append(sc.fit_transform(v))
            scalers.append(sc)
    return scaled, scalers


def apply_scalers(views: List[np.ndarray], scalers) -> List[np.ndarray]:
    """Apply previously-fitted per-view scalers to new data."""
    out = []
    for v, sc in zip(views, scalers):
        out.append(v if sc is None else sc.transform(v))
    return out
