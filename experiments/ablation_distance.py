"""Ablation: distance metric for nearest-class-mean classification."""

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
    p.add_argument("--components", type=int, default=None)
    args = p.parse_args()
    set_seed(args.seed)

    Xtr, Xte, ytr, yte = load_dataset(args)
    mvlda = MultiViewLDA(n_components=args.components, mode=args.mode).fit(Xtr, ytr)

    print(f"Distance-metric ablation ({args.dataset}, mode={args.mode}):\n")
    print(f"{'metric':<14}{'accuracy':<12}")
    print("-" * 26)
    for metric in ["euclidean", "manhattan", "cosine"]:
        pred = NearestClassMean(mvlda, metric=metric).predict(Xte)
        acc = classification_report_from_cm(confusion(yte, pred))["accuracy"]
        print(f"{metric:<14}{acc * 100:<12.3f}")


if __name__ == "__main__":
    main()
