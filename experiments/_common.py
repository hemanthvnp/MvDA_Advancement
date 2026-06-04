"""Shared plumbing for experiment scripts: path setup, data loading, evaluation."""

from __future__ import annotations

import json
import os
import sys
from typing import List, Tuple

import numpy as np

# Make `src/` importable when running scripts directly (no install required).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from mvda.datasets import (  # noqa: E402
    load_colorferet,
    load_multiple_features,
    train_test_split_per_class,
)
from mvda.metrics import classification_report_from_cm, confusion, format_report  # noqa: E402
from mvda.utils import apply_scalers, fit_scalers  # noqa: E402

RESULTS_DIR = os.path.join(_ROOT, "results")


def _split_per_class(views, y, train_frac, seed):
    """Stratified per-class split preserving instance correspondence."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        cut = max(1, int(round(len(idx) * train_frac)))
        train_idx.extend(idx[:cut])
        test_idx.extend(idx[cut:])
    train_idx, test_idx = np.array(train_idx), np.array(test_idx)
    return ([v[train_idx] for v in views], [v[test_idx] for v in views],
            y[train_idx], y[test_idx])


def load_dataset(args):
    """Return (Xtr, Xte, ytr, yte) of scaled views for the chosen dataset."""
    if args.dataset == "mfeat":
        views, y = load_multiple_features(cache_dir=os.path.join(_ROOT, "data", "mfeat"))
        Xtr, Xte, ytr, yte = train_test_split_per_class(views, y, n_train_per_class=100)
    elif args.dataset == "colorferet":
        views, y = load_colorferet(
            root=args.feret_root,
            poses=tuple(args.feret_poses),
            image_size=tuple(args.feret_size),
            max_subjects=args.feret_max_subjects,
            cache_path=os.path.join(_ROOT, "data", "feret_cache.npz"),
        )
        Xtr, Xte, ytr, yte = _split_per_class(views, y, args.train_frac, args.seed)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    Xtr, scalers = fit_scalers(Xtr, args.scaler)
    Xte = apply_scalers(Xte, scalers)
    return Xtr, Xte, ytr, yte


def evaluate(y_true, y_pred, title: str, save_as: str = None, extra: dict = None):
    """Print a confusion-matrix report and optionally persist it as JSON."""
    cm = confusion(y_true, y_pred)
    report = classification_report_from_cm(cm)
    print(f"\n=== {title} ===")
    print(format_report(report, cm))

    if save_as:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        payload = {"title": title, "report": report, "confusion_matrix": cm.tolist()}
        if extra:
            payload.update(extra)
        with open(os.path.join(RESULTS_DIR, save_as), "w") as f:
            json.dump(payload, f, indent=2)
    return report["accuracy"]


def add_data_args(parser):
    """Register the dataset/preprocessing CLI flags shared by all experiments."""
    parser.add_argument("--dataset", choices=["mfeat", "colorferet"], default="mfeat")
    parser.add_argument("--scaler", choices=["robust", "standard", "none"], default="robust")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.5,
                        help="train fraction for colorferet stratified split")
    # ColorFERET-specific
    parser.add_argument("--feret-root", default="data/feret_raw",
                        help="directory with FERET images (local path or mounted Drive)")
    parser.add_argument("--feret-poses", nargs="+", default=["ql", "fa", "qr"])
    parser.add_argument("--feret-size", nargs=2, type=int, default=[64, 64])
    parser.add_argument("--feret-max-subjects", type=int, default=None)
    return parser
