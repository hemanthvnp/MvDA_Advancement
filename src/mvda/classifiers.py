"""Classifiers that operate on the shared MvDA space.

``NearestClassMean`` -- the canonical MvDA decision rule: assign an instance to
    the nearest class centroid in the shared space.

``MvdaEnsemble`` -- the strongest practical configuration found during the
    study: a distance-weighted nearest-neighbour vote on the shared-space
    projection, combined with per-view LDA votes. Reproduces the headline
    accuracy reported in ``docs/FINDINGS.md``.
"""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

from .model import MultiViewLDA


class NearestClassMean:
    """Nearest-class-mean classifier in the shared space of a fitted MultiViewLDA."""

    def __init__(self, mvlda: MultiViewLDA, metric: str = "euclidean") -> None:
        self.mvlda = mvlda
        self.metric = metric
        self.classes_ = mvlda.classes_
        self._mu = np.vstack([mvlda.class_means_[c] for c in self.classes_])

    def predict(self, Xs: List[np.ndarray]) -> np.ndarray:
        Z = self.mvlda.transform(Xs)
        if self.metric == "manhattan":
            d = np.abs(Z[:, None, :] - self._mu[None, :, :]).sum(axis=2)
        elif self.metric == "cosine":
            Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
            Mn = self._mu / (np.linalg.norm(self._mu, axis=1, keepdims=True) + 1e-12)
            d = 1.0 - Zn @ Mn.T
        else:  # euclidean
            d = np.linalg.norm(Z[:, None, :] - self._mu[None, :, :], axis=2)
        return self.classes_[np.argmin(d, axis=1)]


class MvdaEnsemble:
    """Shared-space kNN vote + per-view LDA votes (weighted majority).

    Parameters mirror the best configuration from the experiments: a Manhattan
    1-NN on the shared projection (high weight) plus one LDA per raw view
    (low weight each), combined by weighted voting.
    """

    def __init__(
        self,
        mvlda: MultiViewLDA,
        knn_metric: str = "manhattan",
        knn_k: int = 1,
        knn_weight: float = 2.0,
        view_weight: float = 0.5,
        view_components: int = 9,
    ) -> None:
        self.mvlda = mvlda
        self.knn_metric = knn_metric
        self.knn_k = knn_k
        self.knn_weight = knn_weight
        self.view_weight = view_weight
        self.view_components = view_components
        self.classes_ = mvlda.classes_

    def fit(self, Xs: List[np.ndarray], y: np.ndarray) -> "MvdaEnsemble":
        self._y = np.asarray(y)
        Z = self.mvlda.transform(Xs)
        self._knn = KNeighborsClassifier(n_neighbors=self.knn_k, metric=self.knn_metric)
        self._knn.fit(Z, y)
        self._view_lda = []
        for X in Xs:
            k = min(self.view_components, X.shape[1] - 1, len(self.classes_) - 1)
            lda = LinearDiscriminantAnalysis(n_components=max(1, k))
            lda.fit(X, y)
            self._view_lda.append(lda)
        return self

    def predict(self, Xs: List[np.ndarray]) -> np.ndarray:
        Z = self.mvlda.transform(Xs)
        cls_index = {c: i for i, c in enumerate(self.classes_)}
        votes = np.zeros((Z.shape[0], len(self.classes_)))

        knn_pred = self._knn.predict(Z)
        for i, p in enumerate(knn_pred):
            votes[i, cls_index[p]] += self.knn_weight

        for X, lda in zip(Xs, self._view_lda):
            vp = lda.predict(X)
            for i, p in enumerate(vp):
                votes[i, cls_index[p]] += self.view_weight

        return self.classes_[np.argmax(votes, axis=1)]
