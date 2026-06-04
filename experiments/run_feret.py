"""Cross-pose face recognition on ColorFERET with MvDA.

Each pose is a view and each subject is a class. We keep every image as a
sample, reduce each view with PCA (eigenfaces), learn a shared MvDA subspace
across poses, then classify held-out single-pose images by nearest class mean
in that shared space. This is the canonical "pose as view" MvDA face protocol.

Example
-------
    python experiments/run_feret.py \
        --feret-root /content/drive/MyDrive/colorferet \
        --feret-poses fa fb hl hr --feret-size 64 64 \
        --pca 120 --mode mvda --save feret_mvda.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from mvda import MultiViewLDA, NearestClassMean  # noqa: E402
from mvda.datasets import load_colorferet  # noqa: E402
from mvda.metrics import classification_report_from_cm, confusion, format_report  # noqa: E402
from mvda.utils import set_seed  # noqa: E402


def _per_view_split(X, y, test_frac, rng):
    """Stratified split of one view's samples; classes with a single image go
    entirely to train (nothing to test)."""
    tr, te = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_frac))
        n_test = min(n_test, len(idx) - 1)  # always keep >=1 for training
        te.extend(idx[:n_test])
        tr.extend(idx[n_test:])
    return np.array(tr, dtype=int), np.array(te, dtype=int)


def build_parser():
    p = argparse.ArgumentParser(description="Cross-pose MvDA face recognition on ColorFERET.")
    p.add_argument("--feret-root", default="data/feret_raw")
    p.add_argument("--feret-poses", nargs="+", default=["fa", "fb", "hl", "hr"])
    p.add_argument("--feret-size", nargs=2, type=int, default=[64, 64])
    p.add_argument("--feret-max-subjects", type=int, default=None)
    p.add_argument("--pca", type=int, default=120, help="PCA dims per view (eigenfaces)")
    p.add_argument("--mode", choices=["mvda"], default="mvda",
                   help="only mvda applies (concat needs corresponded views)")
    p.add_argument("--components", type=int, default=None)
    p.add_argument("--ncm-metric", choices=["euclidean", "manhattan", "cosine"], default="cosine")
    p.add_argument("--test-frac", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-cache", action="store_true", help="don't read/write the assembled .npz cache")
    p.add_argument("--save", default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Cache keyed to the assembly params so different configs don't collide.
    tag = f"{'-'.join(args.feret_poses)}_{args.feret_size[0]}x{args.feret_size[1]}_{args.feret_max_subjects}"
    cache_path = None if args.no_cache else os.path.join(_ROOT, "data", f"feret_{tag}.npz")

    print(f"Loading ColorFERET from {args.feret_root} poses={args.feret_poses} ...")
    views, ys = load_colorferet(
        root=args.feret_root,
        poses=tuple(args.feret_poses),
        image_size=tuple(args.feret_size),
        max_subjects=args.feret_max_subjects,
        cache_path=cache_path,
    )
    n_classes = len(np.unique(np.concatenate(ys)))
    print(f"  views={len(views)} samples/view={[len(y) for y in ys]} classes={n_classes}")

    # Per view: stratified split, then StandardScaler + PCA fit on the train part.
    Xtr, ytr, Xte, yte = [], [], [], []
    for v, (X, y) in enumerate(zip(views, ys)):
        tr, te = _per_view_split(X, y, args.test_frac, rng)
        n_pca = min(args.pca, len(tr), X.shape[1])
        scaler = StandardScaler().fit(X[tr])
        pca = PCA(n_components=n_pca, random_state=args.seed).fit(scaler.transform(X[tr]))
        Xtr.append(pca.transform(scaler.transform(X[tr])))
        Xte.append(pca.transform(scaler.transform(X[te])))
        ytr.append(y[tr])
        yte.append(y[te])

    print(f"Fitting MultiViewLDA(mode={args.mode}) on PCA-reduced views "
          f"(dims={[x.shape[1] for x in Xtr]}) ...")
    mvlda = MultiViewLDA(n_components=args.components, mode=args.mode).fit(Xtr, ytr)
    print(f"  shared-space dim = {mvlda.W_.shape[1]}")

    clf = NearestClassMean(mvlda, metric=args.ncm_metric)

    # Evaluate each view's held-out probes, then pool.
    all_true, all_pred = [], []
    print(f"\nPer-pose probe accuracy (NCM {args.ncm_metric}):")
    print(f"{'pose':<8}{'probes':<10}{'accuracy':<12}")
    print("-" * 30)
    for v, pose in enumerate(args.feret_poses):
        if len(yte[v]) == 0:
            print(f"{pose:<8}{0:<10}{'n/a':<12}")
            continue
        pred = clf.predict_view(v, Xte[v])
        acc = (pred == yte[v]).mean()
        print(f"{pose:<8}{len(yte[v]):<10}{acc * 100:<12.3f}")
        all_true.append(yte[v])
        all_pred.append(pred)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    cm = confusion(y_true, y_pred, labels=mvlda.classes_)
    report = classification_report_from_cm(cm)
    print("-" * 30)
    print(f"\n=== ColorFERET | {args.mode} | NCM({args.ncm_metric}) ===")
    print(f"Overall accuracy: {report['accuracy'] * 100:.3f}%  "
          f"({len(y_true)} probes, {n_classes} subjects)")
    print(f"Macro F1: {report['macro']['f1']:.4f}")

    if args.save:
        os.makedirs(os.path.join(_ROOT, "results"), exist_ok=True)
        with open(os.path.join(_ROOT, "results", args.save), "w") as f:
            json.dump({"title": "colorferet", "accuracy": report["accuracy"],
                       "macro_f1": report["macro"]["f1"], "args": vars(args),
                       "n_probes": int(len(y_true)), "n_classes": int(n_classes)}, f, indent=2)
    return report["accuracy"]


if __name__ == "__main__":
    main()
