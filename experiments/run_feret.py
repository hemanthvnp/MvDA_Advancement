"""Cross-pose face recognition on ColorFERET with MvDA.

Each pose is a view and each subject is a class. Two evaluation protocols:

``disjoint`` (the MvDA paper's protocol, Kan et al. 2016) -- train the shared
    subspace on the first ``--train-subjects`` identities (``--images-per-pose``
    images per pose each), then do gallery/probe recognition on the *remaining,
    unseen* identities: one gallery pose provides a reference per test subject,
    and every other-pose image is a probe matched in the shared space.

``closed`` -- learn on a per-view stratified split of all subjects and classify
    held-out single-pose images by nearest class mean.

Each view is reduced by PCA (eigenfaces) before MvDA.

Example (paper protocol, compare solvers)
-----------------------------------------
    for s in ratio exponential harmonic; do
      python experiments/run_feret.py --protocol disjoint --solver $s \
        --feret-root /content/drive/MyDrive/colorferet \
        --feret-poses pl hl ql fa qr hr pr --train-subjects 231 --images-per-pose 4
    done
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
from mvda.datasets import load_colorferet, load_colorferet_grouped  # noqa: E402
from mvda.metrics import classification_report_from_cm, confusion  # noqa: E402
from mvda.utils import set_seed  # noqa: E402


# --------------------------------------------------------------------- helpers
def _fit_view_reducer(X, n_pca, seed):
    """StandardScaler + PCA fitted on one view's training images."""
    n_pca = min(n_pca, X.shape[0], X.shape[1])
    scaler = StandardScaler().fit(X)
    pca = PCA(n_components=n_pca, random_state=seed).fit(scaler.transform(X))
    return scaler, pca


def _reduce(scaler, pca, X):
    return pca.transform(scaler.transform(X))


def _per_view_split(X, y, test_frac, rng):
    tr, te = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_test = min(int(round(len(idx) * test_frac)), len(idx) - 1)
        te.extend(idx[:n_test])
        tr.extend(idx[n_test:])
    return np.array(tr, dtype=int), np.array(te, dtype=int)


# --------------------------------------------------------------------- closed
def run_closed(args, rng):
    cache = None if args.no_cache else os.path.join(
        _ROOT, "data", f"feret_{'-'.join(args.feret_poses)}_{args.feret_size[0]}_{args.feret_max_subjects}.npz")
    views, ys = load_colorferet(args.feret_root, tuple(args.feret_poses),
                                tuple(args.feret_size), max_subjects=args.feret_max_subjects,
                                cache_path=cache)
    n_classes = len(np.unique(np.concatenate(ys)))
    print(f"  views={len(views)} samples/view={[len(y) for y in ys]} classes={n_classes}")

    Xtr, ytr, Xte, yte = [], [], [], []
    for X, y in zip(views, ys):
        tr, te = _per_view_split(X, y, args.test_frac, rng)
        scaler, pca = _fit_view_reducer(X[tr], args.pca, args.seed)
        Xtr.append(_reduce(scaler, pca, X[tr])); ytr.append(y[tr])
        Xte.append(_reduce(scaler, pca, X[te])); yte.append(y[te])

    mvlda = MultiViewLDA(n_components=args.components, mode="mvda", solver=args.solver).fit(Xtr, ytr)
    clf = NearestClassMean(mvlda, metric=args.ncm_metric)
    all_true, all_pred = [], []
    print(f"\nPer-pose probe accuracy (NCM {args.ncm_metric}):\n{'pose':<8}{'probes':<10}{'acc':<10}")
    print("-" * 28)
    for v, pose in enumerate(args.feret_poses):
        if len(yte[v]) == 0:
            continue
        pred = clf.predict_view(v, Xte[v])
        print(f"{pose:<8}{len(yte[v]):<10}{(pred == yte[v]).mean() * 100:<10.2f}")
        all_true.append(yte[v]); all_pred.append(pred)
    return np.concatenate(all_true), np.concatenate(all_pred), mvlda.classes_, n_classes


# ------------------------------------------------------------------- disjoint
def run_disjoint(args, rng):
    cache = None if args.no_cache else os.path.join(
        _ROOT, "data", f"feretg_{'-'.join(args.feret_poses)}_{args.feret_size[0]}_{args.feret_max_subjects}.npz")
    groups, poses = load_colorferet_grouped(args.feret_root, tuple(args.feret_poses),
                                            tuple(args.feret_size), max_subjects=args.feret_max_subjects,
                                            cache_path=cache)
    labels = sorted(groups)
    if args.train_subjects >= len(labels):
        raise SystemExit(f"train-subjects ({args.train_subjects}) >= available subjects ({len(labels)})")
    train_labels = labels[:args.train_subjects]
    test_labels = labels[args.train_subjects:]
    if args.gallery_pose not in poses:
        raise SystemExit(f"gallery pose {args.gallery_pose} not in {poses}")
    g_idx = poses.index(args.gallery_pose)
    print(f"  subjects: {len(train_labels)} train / {len(test_labels)} test; "
          f"poses={poses}; gallery={args.gallery_pose}")

    # --- build training per-view sets (sample images_per_pose per pose) ---
    Xtr = [[] for _ in poses]
    ytr = [[] for _ in poses]
    for lab in train_labels:
        for v, pose in enumerate(poses):
            imgs = groups[lab][pose]
            if args.images_per_pose and len(imgs) > args.images_per_pose:
                imgs = imgs[rng.choice(len(imgs), args.images_per_pose, replace=False)]
            Xtr[v].append(imgs); ytr[v].append(np.full(len(imgs), lab))
    Xtr = [np.vstack(x) for x in Xtr]
    ytr = [np.concatenate(y) for y in ytr]

    reducers = [_fit_view_reducer(Xtr[v], args.pca, args.seed) for v in range(len(poses))]
    Xtr = [_reduce(*reducers[v], Xtr[v]) for v in range(len(poses))]

    mvlda = MultiViewLDA(n_components=args.components, mode="mvda", solver=args.solver).fit(Xtr, ytr)
    print(f"  shared-space dim = {mvlda.W_.shape[1]}")

    # --- gallery (mean of gallery-pose images per test subject) ---
    gal_refs, gal_labels = [], []
    for lab in test_labels:
        z = mvlda.transform_view(g_idx, _reduce(*reducers[g_idx], groups[lab][args.gallery_pose]))
        gal_refs.append(z.mean(axis=0)); gal_labels.append(lab)
    gal_refs = np.vstack(gal_refs)
    gal_labels = np.array(gal_labels)
    gal_n = gal_refs / (np.linalg.norm(gal_refs, axis=1, keepdims=True) + 1e-12)

    # --- probes: every non-gallery-pose image of each test subject ---
    all_true, all_pred = [], []
    print(f"\nPer-pose probe rank-1 accuracy:\n{'pose':<8}{'probes':<10}{'acc':<10}")
    print("-" * 28)
    for v, pose in enumerate(poses):
        if v == g_idx:
            continue
        Xp, yp = [], []
        for lab in test_labels:
            Xp.append(groups[lab][pose]); yp.append(np.full(len(groups[lab][pose]), lab))
        Xp = np.vstack(Xp); yp = np.concatenate(yp)
        Z = mvlda.transform_view(v, _reduce(*reducers[v], Xp))
        Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
        pred = gal_labels[np.argmax(Zn @ gal_n.T, axis=1)]
        print(f"{pose:<8}{len(yp):<10}{(pred == yp).mean() * 100:<10.2f}")
        all_true.append(yp); all_pred.append(pred)
    return np.concatenate(all_true), np.concatenate(all_pred), np.array(test_labels), len(test_labels)


# ----------------------------------------------------------------------- main
def build_parser():
    p = argparse.ArgumentParser(description="Cross-pose MvDA face recognition on ColorFERET.")
    p.add_argument("--feret-root", default="data/feret_raw")
    p.add_argument("--feret-poses", nargs="+", default=["pl", "hl", "ql", "fa", "qr", "hr", "pr"])
    p.add_argument("--feret-size", nargs=2, type=int, default=[64, 64])
    p.add_argument("--feret-max-subjects", type=int, default=None)
    p.add_argument("--protocol", choices=["disjoint", "closed"], default="disjoint",
                   help="disjoint=MvDA-paper gallery/probe on unseen IDs; closed=per-view split")
    p.add_argument("--train-subjects", type=int, default=231, help="disjoint: #identities for training")
    p.add_argument("--images-per-pose", type=int, default=4, help="disjoint: train images/pose/subject")
    p.add_argument("--gallery-pose", default="fa", help="disjoint: pose used as the gallery")
    p.add_argument("--pca", type=int, default=120, help="PCA dims per view (eigenfaces)")
    p.add_argument("--solver", choices=["ratio", "exponential", "harmonic"], default="ratio")
    p.add_argument("--components", type=int, default=None)
    p.add_argument("--ncm-metric", choices=["euclidean", "manhattan", "cosine"], default="cosine")
    p.add_argument("--test-frac", type=float, default=0.4, help="closed protocol only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--save", default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"ColorFERET [{args.protocol}] root={args.feret_root} poses={args.feret_poses} "
          f"solver={args.solver} ...")
    runner = run_disjoint if args.protocol == "disjoint" else run_closed
    y_true, y_pred, classes, n_classes = runner(args, rng)

    cm = confusion(y_true, y_pred, labels=classes)
    report = classification_report_from_cm(cm)
    print("-" * 28)
    print(f"\n=== ColorFERET | {args.protocol} | {args.solver} | rank-1 ===")
    print(f"Overall accuracy: {report['accuracy'] * 100:.3f}%  "
          f"({len(y_true)} probes, {n_classes} subjects)")
    print(f"Macro F1: {report['macro']['f1']:.4f}")

    if args.save:
        os.makedirs(os.path.join(_ROOT, "results"), exist_ok=True)
        with open(os.path.join(_ROOT, "results", args.save), "w") as f:
            json.dump({"protocol": args.protocol, "solver": args.solver,
                       "accuracy": report["accuracy"], "macro_f1": report["macro"]["f1"],
                       "n_probes": int(len(y_true)), "n_subjects": int(n_classes),
                       "args": vars(args)}, f, indent=2)
    return report["accuracy"]


if __name__ == "__main__":
    main()
