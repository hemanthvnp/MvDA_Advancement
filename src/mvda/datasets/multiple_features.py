"""UCI Multiple Features dataset (a.k.a. mfeat).

2000 handwritten digit instances (200 per class, 10 classes), each described by
six feature sets ("views"):

    fou  76  Fourier coefficients of the character shapes
    fac  216 profile correlations
    kar  64  Karhunen-Loeve coefficients
    pix  240 pixel averages in 2x3 windows
    zer  47  Zernike moments
    mor  6   morphological features

This is a natural multi-view benchmark: the same digit instance is described six
different ways, giving genuine instance correspondence across views.

Reference: https://archive.ics.uci.edu/dataset/72/multiple+features
"""

from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from typing import List, Tuple

import numpy as np

UCI_URL = "https://archive.ics.uci.edu/static/public/72/multiple+features.zip"
VIEW_FILES = ["mfeat-fou", "mfeat-fac", "mfeat-kar", "mfeat-pix", "mfeat-zer", "mfeat-mor"]
VIEW_NAMES = ["fou", "fac", "kar", "pix", "zer", "mor"]


def load_multiple_features(cache_dir: str = "data/mfeat") -> Tuple[List[np.ndarray], np.ndarray]:
    """Download (once) and load the six views. Returns ``(views, y)``.

    The archive is cached under ``cache_dir`` so subsequent runs are offline.
    Labels are ``0..9`` repeated per the dataset's fixed ordering (200 per class,
    contiguous blocks).
    """
    os.makedirs(cache_dir, exist_ok=True)
    archive = os.path.join(cache_dir, "multiple_features.zip")
    if not os.path.exists(archive):
        with urllib.request.urlopen(UCI_URL) as r:
            data = r.read()
        with open(archive, "wb") as f:
            f.write(data)

    with zipfile.ZipFile(archive) as z:
        views = [np.loadtxt(z.open(name)) for name in VIEW_FILES]

    # The dataset stores 2000 rows ordered as 200 of class 0, 200 of class 1, ...
    y = np.repeat(np.arange(10), 200)
    return views, y


def train_test_split_per_class(
    views: List[np.ndarray],
    y: np.ndarray,
    n_train_per_class: int = 100,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    """Deterministic per-class split preserving instance correspondence.

    The canonical mfeat protocol uses the first ``n_train_per_class`` instances of
    each class for training and the rest for testing.
    """
    classes = np.unique(y)
    train_idx, test_idx = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        train_idx.extend(idx[:n_train_per_class])
        test_idx.extend(idx[n_train_per_class:])
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    Xtr = [v[train_idx] for v in views]
    Xte = [v[test_idx] for v in views]
    return Xtr, Xte, y[train_idx], y[test_idx]
