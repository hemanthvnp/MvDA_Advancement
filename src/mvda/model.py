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
        return [X.shape[0] for X in Xs]

    def _check_corresponded(self, Xs: List[np.ndarray]):
        """Validate the shared-label contract: equal rows across views."""
        ns = self._check(Xs)
        if len(set(ns)) != 1:
            raise ValueError(
                "this operation needs corresponded views (equal rows per view); "
                "got sizes " + str(ns)
            )
        return ns[0]

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

    def _embed(self, Xs: List[np.ndarray], ys: List[np.ndarray]):
        """Block-embed each view-sample into the stacked feature space.

        Each view may carry a different number of (independently labelled)
        samples; row blocks are concatenated and each sits in its own view-block.
        """
        total_rows = int(sum(X.shape[0] for X in Xs))
        Phi = np.zeros((total_rows, self.total_dim_))
        labels = np.empty(total_rows, dtype=int)
        r = 0
        for v, (X, yv) in enumerate(zip(Xs, ys)):
            nv = X.shape[0]
            Phi[r:r + nv, self.blocks_[v]:self.blocks_[v + 1]] = X
            labels[r:r + nv] = yv
            r += nv
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
    def fit(self, Xs: List[np.ndarray], y) -> "MultiViewLDA":
        """Fit the shared subspace.

        ``y`` may be either:
          * a single label array (length = rows per view) -- the *corresponded*
            contract: row ``i`` of every view is the same instance. Required for
            ``mode="concat"`` and for view-consistency.
          * a list of per-view label arrays -- views may have different sample
            counts and no row correspondence; classes are pooled by label.
        """
        ns = self._check(Xs)
        self.dims_ = [X.shape[1] for X in Xs]
        self.total_dim_ = int(sum(self.dims_))
        self.n_views_ = len(Xs)
        self.blocks_ = np.cumsum([0] + self.dims_)

        per_view_labels = isinstance(y, (list, tuple))
        if per_view_labels:
            ys = [np.asarray(a) for a in y]
            if len(ys) != self.n_views_ or any(len(a) != n for a, n in zip(ys, ns)):
                raise ValueError("per-view labels must match each view's row count")
            self._y = None
        else:
            self._check_corresponded(Xs)
            self._y = np.asarray(y)
            if self._y.shape[0] != ns[0]:
                raise ValueError("y must have one label per instance row")
            ys = [self._y] * self.n_views_

        if self.mode == "concat" and per_view_labels:
            raise ValueError("mode='concat' requires corresponded views with a single label array")
        if self.vc_lambda and per_view_labels:
            raise ValueError("view-consistency requires corresponded views")

        self.classes_ = np.unique(np.concatenate(ys))
        k = self.n_components or (len(self.classes_) - 1)
        k = min(k, len(self.classes_) - 1, self.total_dim_)

        if self.mode == "concat":
            Phi, labels = np.hstack(Xs), self._y
        else:
            Phi, labels = self._embed(Xs, ys)

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

        # Shared-space class centroids (projected mean of each class's embedded
        # samples). Used by nearest-class-mean classification.
        means_emb = np.vstack([Phi[labels == c].mean(axis=0) for c in self.classes_])
        proj = means_emb @ self.W_
        self.class_means_ = {c: proj[i] for i, c in enumerate(self.classes_)}
        return self

    # -------------------------------------------------------------- transform
    def transform(self, Xs: List[np.ndarray]) -> np.ndarray:
        """Project a *corresponded* multi-view instance set into the shared space.

        ``concat``: stacked features times W (= sum over views).
        ``mvda``  : average of the per-view projections (the natural shared-space
                    location of an instance seen through every view).

        Requires equal rows per view (one instance seen through all views).
        For independently-sampled views use :meth:`transform_view`.
        """
        self._check_corresponded(Xs)
        if self.mode == "concat":
            return np.hstack(Xs) @ self.W_
        acc = np.zeros((Xs[0].shape[0], self.W_.shape[1]))
        for v, X in enumerate(Xs):
            acc += X @ self.W_views_[v]
        return acc / self.n_views_

    def transform_view(self, v: int, X: np.ndarray) -> np.ndarray:
        """Project samples seen only through view ``v`` into the shared space."""
        return X @ self.W_views_[v]

    def transform_views(self, Xs: List[np.ndarray]) -> List[np.ndarray]:
        """Per-view projections into the shared space (list of (n_v, k) arrays)."""
        self._check(Xs)
        return [self.transform_view(v, X) for v, X in enumerate(Xs)]

    def fit_transform(self, Xs: List[np.ndarray], y) -> np.ndarray:
        return self.fit(Xs, y).transform(Xs)
