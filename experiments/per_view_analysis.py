"""Per-view diagnostic: how discriminative is each view on its own?

Fits an ordinary LDA on each individual view and reports its test accuracy,
illustrating the spread in view quality that multi-view fusion exploits.
"""

from __future__ import annotations

import argparse
import os
import sys

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import add_data_args, load_dataset  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from mvda import MultiViewLDA, NearestClassMean  # noqa: E402
from mvda.metrics import classification_report_from_cm, confusion  # noqa: E402
from mvda.utils import set_seed  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    add_data_args(p)
    args = p.parse_args()
    set_seed(args.seed)

    Xtr, Xte, ytr, yte = load_dataset(args)
    n_classes = len(set(ytr))

    print(f"Per-view LDA accuracy ({args.dataset}):\n")
    print(f"{'view':<8}{'dim':<8}{'accuracy':<12}")
    print("-" * 28)
    for v in range(len(Xtr)):
        k = min(n_classes - 1, Xtr[v].shape[1] - 1)
        lda = LinearDiscriminantAnalysis(n_components=max(1, k)).fit(Xtr[v], ytr)
        acc = (lda.predict(Xte[v]) == yte).mean()
        print(f"{v:<8}{Xtr[v].shape[1]:<8}{acc * 100:<12.3f}")

    mvlda = MultiViewLDA(mode="mvda").fit(Xtr, ytr)
    pred = NearestClassMean(mvlda, metric="cosine").predict(Xte)
    fused = classification_report_from_cm(confusion(yte, pred))["accuracy"]
    print("-" * 28)
    print(f"{'MvDA':<8}{'fused':<8}{fused * 100:<12.3f}")


if __name__ == "__main__":
    main()
