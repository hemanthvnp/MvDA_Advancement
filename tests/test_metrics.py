"""Unit tests for confusion-matrix-derived metrics."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from mvda.metrics import classification_report_from_cm  # noqa: E402


def test_perfect_classifier():
    cm = np.array([[10, 0], [0, 10]])
    rep = classification_report_from_cm(cm)
    assert rep["accuracy"] == 1.0
    assert rep["macro"]["f1"] == 1.0


def test_known_values():
    # class 0: 8 TP, 2 FN; class 1: 1 FP, 9 TP
    cm = np.array([[8, 2], [1, 9]])
    rep = classification_report_from_cm(cm)
    assert abs(rep["accuracy"] - 17 / 20) < 1e-9
    # precision_0 = 8 / (8 + 1) = 0.888...
    assert abs(rep["per_class"]["precision"][0] - 8 / 9) < 1e-9
    # recall_0 = 8 / (8 + 2) = 0.8
    assert abs(rep["per_class"]["recall"][0] - 0.8) < 1e-9


def test_empty_class_safe():
    cm = np.array([[5, 0], [0, 0]])  # no samples of class 1
    rep = classification_report_from_cm(cm)
    assert rep["accuracy"] == 1.0
    assert np.isfinite(rep["macro"]["f1"])
