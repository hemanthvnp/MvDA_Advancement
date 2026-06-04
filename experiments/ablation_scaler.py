"""Ablation: per-view preprocessing (scaler choice)."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import _ROOT  # noqa: E402

sys.path.insert(0, os.path.join(_ROOT, "src"))
from mvda import MultiViewLDA, MvdaEnsemble  # noqa: E402
from mvda.datasets import load_multiple_features, train_test_split_per_class  # noqa: E402
from mvda.utils import apply_scalers, fit_scalers, set_seed  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["mvda", "concat"], default="concat")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    set_seed(args.seed)

    views, y = load_multiple_features(cache_dir=os.path.join(_ROOT, "data", "mfeat"))
    Xtr0, Xte0, ytr, yte = train_test_split_per_class(views, y, n_train_per_class=100)

    print(f"Scaler ablation (mfeat, mode={args.mode}, ensemble):\n")
    print(f"{'scaler':<12}{'accuracy':<12}")
    print("-" * 24)
    for scaler in ["none", "standard", "robust"]:
        Xtr, sc = fit_scalers(Xtr0, scaler)
        Xte = apply_scalers(Xte0, sc)
        mvlda = MultiViewLDA(mode=args.mode).fit(Xtr, ytr)
        clf = MvdaEnsemble(mvlda).fit(Xtr, ytr)
        acc = (clf.predict(Xte) == yte).mean()
        print(f"{scaler:<12}{acc * 100:<12.3f}")


if __name__ == "__main__":
    main()
