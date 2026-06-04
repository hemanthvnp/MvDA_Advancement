"""Ablation: discriminant solver (classical ratio vs exponential vs harmonic).

- ratio:       classical LDA generalized eigenproblem.
- exponential: Exponential DA -- exp(S_b) w = lambda exp(S_w) w (robust to the
               small-sample-size singularity; enlarges margins).
- harmonic:    Harmonic-mean LDA -- reweights pairwise between-class scatter to
               emphasize close (confusable) class pairs.

On the near-saturated UCI digits (n >> d) the classical solver is already at
ceiling; the exponential/harmonic variants are designed to help in the
high-dimensional small-sample regime (e.g. ColorFERET eigenfaces), so benchmark
them there too via run_feret.py --solver ...
"""

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
    args = p.parse_args()
    set_seed(args.seed)

    Xtr, Xte, ytr, yte = load_dataset(args)
    print(f"Solver ablation ({args.dataset}, mode={args.mode}, NCM cosine):\n")
    print(f"{'solver':<14}{'accuracy':<12}")
    print("-" * 26)
    for solver in ["ratio", "exponential", "harmonic"]:
        mvlda = MultiViewLDA(mode=args.mode, solver=solver).fit(Xtr, ytr)
        pred = NearestClassMean(mvlda, metric="cosine").predict(Xte)
        acc = classification_report_from_cm(confusion(yte, pred))["accuracy"]
        print(f"{solver:<14}{acc * 100:<12.3f}")


if __name__ == "__main__":
    main()
