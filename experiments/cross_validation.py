"""K-fold cross-validation of the full MvDA pipeline.

Reports mean +/- std accuracy across folds, a more robust estimate than the
single canonical hold-out split.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import add_data_args  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from mvda import MultiViewLDA, MvdaEnsemble, NearestClassMean  # noqa: E402
from mvda.datasets import load_colorferet, load_multiple_features  # noqa: E402
from mvda.utils import apply_scalers, fit_scalers, set_seed  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_raw(args):
    if args.dataset == "mfeat":
        return load_multiple_features(cache_dir=os.path.join(_ROOT, "data", "mfeat"))
    return load_colorferet(root=args.feret_root, poses=tuple(args.feret_poses),
                           image_size=tuple(args.feret_size),
                           max_subjects=args.feret_max_subjects,
                           cache_path=os.path.join(_ROOT, "data", "feret_cache.npz"))


def main():
    p = argparse.ArgumentParser()
    add_data_args(p)
    p.add_argument("--mode", choices=["mvda", "concat"], default="concat")
    p.add_argument("--classifier", choices=["ncm", "ensemble"], default="ensemble")
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args()
    set_seed(args.seed)

    views, y = _load_raw(args)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    scores = []
    for fold, (tr, te) in enumerate(skf.split(views[0], y), 1):
        Xtr = [v[tr] for v in views]
        Xte = [v[te] for v in views]
        Xtr, sc = fit_scalers(Xtr, args.scaler)
        Xte = apply_scalers(Xte, sc)

        mvlda = MultiViewLDA(mode=args.mode).fit(Xtr, y[tr])
        if args.classifier == "ensemble":
            clf = MvdaEnsemble(mvlda).fit(Xtr, y[tr])
        else:
            clf = NearestClassMean(mvlda, metric="cosine")
        acc = (clf.predict(Xte) == y[te]).mean()
        scores.append(acc)
        print(f"Fold {fold}: {acc * 100:.3f}%")

    scores = np.array(scores)
    print(f"\n{args.folds}-fold CV: {scores.mean() * 100:.3f}% +/- {scores.std() * 100:.3f}%")


if __name__ == "__main__":
    main()
