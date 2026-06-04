"""Multi-view Linear Discriminant Analysis.

Implements two related linear projections that map several *views* of the same
set of instances into a single, shared, low-dimensional discriminative space.

``mode="mvda"`` -- Genuine Multi-view Discriminant Analysis (Kan et al.,
    "Multi-view Discriminant Analysis", IEEE TPAMI 2016). Each view ``v`` is
    given its own linear transform ``W_v``. Between- and within-class scatter
    are pooled across *all* views, so the class structure is shared while each
    view keeps its own projection. We solve this with a block-embedding trick:
    every view-sample is embedded into a block-sparse vector (its features sit
    in its own view-block, zeros elsewhere), and ordinary LDA on the stacked
    embedded samples is *exactly* the MvDA generalized eigenproblem. This makes
    the implementation short and obviously correct.

``mode="concat"`` -- Concatenation-LDA baseline: stack the views into one long
    feature vector and run ordinary LDA. Requires sample correspondence across
    views and treats the views as one feature space.

An optional view-consistency (VC) penalty (``vc_lambda``) encourages different
views' projections of the *same* instance to land close together.

Data contract: ``fit(Xs, y)`` takes ``Xs`` = list of ``V`` arrays each shaped
``(n, d_v)``; row ``i`` of every view is the same instance, with label ``y[i]``.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.linalg import eigh


class MultiViewLDA:
    def __init__(
        self,
        n_components: Optional[int] = None,
        mode: str = "mvda",
        vc_lambda: float = 0.0,
        reg: float = 1e-6,
    ) -> None:
        if mode not in {"mvda", "concat"}:
            raise ValueError("mode must be 'mvda' or 'concat'")
        if mode == "concat" and vc_lambda:
            raise ValueError("view-consistency is only defined for mode='mvda'")
        self.n_components = n_components
        self.mode = mode
        self.vc_lambda = vc_lambda
        self.reg = reg

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _check(Xs: List[np.ndarray]):
        if len(Xs) < 2:
            raise ValueError("MvDA needs at least two views")
        n = Xs[0].shape[0]
        if any(X.shape[0] != n for X in Xs):
            raise ValueError("all views must share the same number of rows (instance correspondence)")
        return n

    def _scatter(self, Phi: np.ndarray, labels: np.ndarray):
        """Standard between/within-class scatter of (already embedded) samples."""
        D = Phi.shape[1]
        classes = np.unique(labels)
        mu = Phi.mean(axis=0)
        S_w = np.zeros((D, D))
        S_b = np.zeros((D, D))
        for c in classes:
            Xc = Phi[labels == c]
            mu_c = Xc.mean(axis=0)
            diff = Xc - mu_c
            S_w += diff.T @ diff
            d = (mu_c - mu).reshape(-1, 1)
            S_b += Xc.shape[0] * (d @ d.T)
        return S_b, S_w

    def _embed(self, Xs: List[np.ndarray]):
        """Block-embed each view-sample into the stacked feature space."""
        n = Xs[0].shape[0]
        Phi = np.zeros((self.n_views_ * n, self.total_dim_))
        labels = np.empty(self.n_views_ * n, dtype=int)
        for v, X in enumerate(Xs):
            rows = slice(v * n, (v + 1) * n)
            Phi[rows, self.blocks_[v]:self.blocks_[v + 1]] = X
            labels[rows] = self._y
        return Phi, labels

    def _vc_scatter(self, Xs: List[np.ndarray]) -> np.ndarray:
        """View-consistency scatter: sum over instances & view-pairs of the
        embedded (W_u^T x_iu - W_v^T x_iv) outer products (pre-projection)."""
        n = Xs[0].shape[0]
        diffs = []
        for u in range(self.n_views_):
            for w in range(u + 1, self.n_views_):
                blk = np.zeros((n, self.total_dim_))
                blk[:, self.blocks_[u]:self.blocks_[u + 1]] = Xs[u]
                blk[:, self.blocks_[w]:self.blocks_[w + 1]] = -Xs[w]
                diffs.append(blk)
        Diff = np.vstack(diffs)
        return Diff.T @ Diff

    # -------------------------------------------------------------------- fit
    def fit(self, Xs: List[np.ndarray], y: np.ndarray) -> "MultiViewLDA":
        n = self._check(Xs)
        y = np.asarray(y)
        if y.shape[0] != n:
            raise ValueError("y must have one label per instance row")
        self._y = y
        self.classes_ = np.unique(y)
        self.dims_ = [X.shape[1] for X in Xs]
        self.total_dim_ = int(sum(self.dims_))
        self.n_views_ = len(Xs)
        self.blocks_ = np.cumsum([0] + self.dims_)

        k = self.n_components or (len(self.classes_) - 1)
        k = min(k, len(self.classes_) - 1, self.total_dim_)

        if self.mode == "concat":
            Phi = np.hstack(Xs)
            labels = y
        else:
            Phi, labels = self._embed(Xs)

        S_b, S_w = self._scatter(Phi, labels)
        if self.vc_lambda:
            S_w = S_w + self.vc_lambda * self._vc_scatter(Xs)

        # Regularize within-class scatter for a stable generalized eigenproblem.
        scale = max(1e-12, np.trace(S_w) / S_w.shape[0])
        S_w_reg = S_w + max(1e-8, scale * self.reg) * np.eye(S_w.shape[0])

        try:
            eigvals, eigvecs = eigh(S_b, S_w_reg)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(S_w_reg) @ S_b)
        eigvals, eigvecs = np.real(eigvals), np.real(eigvecs)

        order = np.argsort(-eigvals)[:k]
        self.eigenvalues_ = eigvals[order]
        self.W_ = eigvecs[:, order]                      # (D, k) stacked transform
        self.W_views_ = [self.W_[self.blocks_[v]:self.blocks_[v + 1]] for v in range(self.n_views_)]

        # Class centroids in the shared space (for nearest-class-mean classify).
        Z = self.transform(Xs)
        self.class_means_ = {c: Z[y == c].mean(axis=0) for c in self.classes_}
        return self

    # -------------------------------------------------------------- transform
    def transform(self, Xs: List[np.ndarray]) -> np.ndarray:
        """Project a multi-view instance set into the shared space.

        ``concat``: stacked features times W (= sum over views).
        ``mvda``  : average of the per-view projections (the natural shared-space
                    location of an instance seen through every view).
        """
        self._check(Xs)
        if self.mode == "concat":
            return np.hstack(Xs) @ self.W_
        acc = np.zeros((Xs[0].shape[0], self.W_.shape[1]))
        for v, X in enumerate(Xs):
            acc += X @ self.W_views_[v]
        return acc / self.n_views_

    def transform_views(self, Xs: List[np.ndarray]) -> List[np.ndarray]:
        """Per-view projections into the shared space (list of (n, k) arrays)."""
        self._check(Xs)
        return [X @ self.W_views_[v] for v, X in enumerate(Xs)]

    def fit_transform(self, Xs: List[np.ndarray], y: np.ndarray) -> np.ndarray:
        return self.fit(Xs, y).transform(Xs)
