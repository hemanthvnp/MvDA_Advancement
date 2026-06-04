"""Multi-view Discriminant Analysis (MvDA).

A small, dependency-light implementation of Multi-view Discriminant Analysis
(Kan et al., IEEE TPAMI 2016) together with a concatenation-LDA baseline,
nearest-class-mean / ensemble classifiers, and reproducible experiment
runners over the UCI Multiple Features and ColorFERET datasets.
"""

from .model import MultiViewLDA
from .classifiers import NearestClassMean, MvdaEnsemble
from .metrics import classification_report_from_cm, confusion

__all__ = [
    "MultiViewLDA",
    "NearestClassMean",
    "MvdaEnsemble",
    "classification_report_from_cm",
    "confusion",
]

__version__ = "0.1.0"
