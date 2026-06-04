"""Main experiment: fit MvDA and evaluate a classifier on the shared space.

Examples
--------
    # genuine MvDA + nearest-class-mean on UCI Multiple Features
    python experiments/run_mvda.py --mode mvda --classifier ncm

    # concatenation-LDA baseline + the weighted ensemble (headline config)
    python experiments/run_mvda.py --mode concat --classifier ensemble

    # MvDA with view-consistency on ColorFERET (Drive/local images)
    python experiments/run_mvda.py --dataset colorferet --mode mvda --vc-lambda 0.05 \
        --feret-root /content/drive/MyDrive/colorferet
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import add_data_args, evaluate, load_dataset  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from mvda import MultiViewLDA, MvdaEnsemble, NearestClassMean  # noqa: E402
from mvda.utils import set_seed  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description="Fit MvDA and evaluate on the shared space.")
    add_data_args(p)
    p.add_argument("--mode", choices=["mvda", "concat"], default="mvda")
    p.add_argument("--solver", choices=["ratio", "exponential", "harmonic"], default="ratio",
                   help="ratio=classical LDA; exponential=EDA; harmonic=HM-LDA")
    p.add_argument("--classifier", choices=["ncm", "ensemble"], default="ncm")
    p.add_argument("--components", type=int, default=None,
                   help="shared-space dimension (default C-1)")
    p.add_argument("--vc-lambda", type=float, default=0.0,
                   help="view-consistency strength (mode=mvda only)")
    p.add_argument("--ncm-metric", choices=["euclidean", "manhattan", "cosine"],
                   default="cosine")
    p.add_argument("--save", default=None, help="filename under results/ to save JSON")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    set_seed(args.seed)

    print(f"Loading dataset={args.dataset} scaler={args.scaler} ...")
    Xtr, Xte, ytr, yte = load_dataset(args)
    print(f"  views={len(Xtr)} dims={[v.shape[1] for v in Xtr]} "
          f"train={Xtr[0].shape[0]} test={Xte[0].shape[0]} classes={len(set(ytr))}")

    print(f"Fitting MultiViewLDA(mode={args.mode}, vc_lambda={args.vc_lambda}) ...")
    mvlda = MultiViewLDA(n_components=args.components, mode=args.mode,
                         vc_lambda=args.vc_lambda, solver=args.solver)
    mvlda.fit(Xtr, ytr)
    print(f"  shared-space dim = {mvlda.W_.shape[1]}")

    if args.classifier == "ncm":
        clf = NearestClassMean(mvlda, metric=args.ncm_metric)
        title = f"{args.dataset} | {args.mode} | NCM({args.ncm_metric})"
    else:
        clf = MvdaEnsemble(mvlda).fit(Xtr, ytr)
        title = f"{args.dataset} | {args.mode} | ensemble"

    y_pred = clf.predict(Xte)
    acc = evaluate(yte, y_pred, title, save_as=args.save,
                   extra={"args": vars(args)})
    print(f"\nFinal accuracy: {acc * 100:.3f}%")
    return acc


if __name__ == "__main__":
    main()
