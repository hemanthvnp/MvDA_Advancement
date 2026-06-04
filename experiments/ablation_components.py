"""Ablation: shared-space dimensionality (number of components)."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import add_data_args, load_dataset  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from mvda import MultiViewLDA, NearestClassMean  # noqa: E402
from mvda.metrics import classification_report_from_cm, confusion  # noqa: E402
from mvda.utils import set_seed  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    add_data_args(p)
    p.add_argument("--mode", choices=["mvda", "concat"], default="mvda")
    p.add_argument("--max-components", type=int, default=9)
    args = p.parse_args()
    set_seed(args.seed)

    Xtr, Xte, ytr, yte = load_dataset(args)
    print(f"Component sweep ({args.dataset}, mode={args.mode}):\n")
    print(f"{'k':<6}{'accuracy':<12}")
    print("-" * 18)
    best = (0, 0.0)
    for k in range(1, args.max_components + 1):
        mvlda = MultiViewLDA(n_components=k, mode=args.mode).fit(Xtr, ytr)
        pred = NearestClassMean(mvlda, metric="cosine").predict(Xte)
        acc = classification_report_from_cm(confusion(yte, pred))["accuracy"]
        print(f"{k:<6}{acc * 100:<12.3f}")
        if acc > best[1]:
            best = (k, acc)
    print(f"\nBest: k={best[0]} -> {best[1] * 100:.3f}%")


if __name__ == "__main__":
    main()
