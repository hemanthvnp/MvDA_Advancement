"""Evaluation metrics derived from the confusion matrix."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import confusion_matrix


def confusion(y_true: np.ndarray, y_pred: np.ndarray, labels=None) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)


def classification_report_from_cm(cm: np.ndarray) -> Dict:
    """Compute accuracy and per-class precision/recall/F1 from a confusion matrix.

    Everything is derived from ``cm`` so the report is internally consistent with
    the matrix printed alongside it.
    """
    cm = np.asarray(cm, dtype=float)
    n_classes = cm.shape[0]
    total = cm.sum()
    accuracy = np.trace(cm) / total if total else 0.0

    precision, recall, f1, support = [], [], [], []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(f)
        support.append(cm[c, :].sum())

    support = np.array(support)
    w = support / support.sum() if support.sum() else np.ones(n_classes) / n_classes
    return {
        "accuracy": float(accuracy),
        "per_class": {
            "precision": [float(x) for x in precision],
            "recall": [float(x) for x in recall],
            "f1": [float(x) for x in f1],
            "support": [int(x) for x in support],
        },
        "macro": {
            "precision": float(np.mean(precision)),
            "recall": float(np.mean(recall)),
            "f1": float(np.mean(f1)),
        },
        "weighted": {
            "precision": float(np.average(precision, weights=w)),
            "recall": float(np.average(recall, weights=w)),
            "f1": float(np.average(f1, weights=w)),
        },
    }


def format_report(report: Dict, cm: np.ndarray) -> str:
    """Human-readable rendering of a report + confusion matrix."""
    lines = []
    pc = report["per_class"]
    n = len(pc["precision"])
    lines.append(f"Accuracy: {report['accuracy'] * 100:.3f}%")
    lines.append("")
    lines.append(f"{'class':<8}{'precision':<12}{'recall':<12}{'f1':<12}{'support':<10}")
    lines.append("-" * 54)
    for c in range(n):
        lines.append(
            f"{c:<8}{pc['precision'][c]:<12.4f}{pc['recall'][c]:<12.4f}"
            f"{pc['f1'][c]:<12.4f}{pc['support'][c]:<10}"
        )
    lines.append("-" * 54)
    m, wd = report["macro"], report["weighted"]
    lines.append(f"{'macro':<8}{m['precision']:<12.4f}{m['recall']:<12.4f}{m['f1']:<12.4f}")
    lines.append(f"{'weighted':<8}{wd['precision']:<12.4f}{wd['recall']:<12.4f}{wd['f1']:<12.4f}")
    return "\n".join(lines)
