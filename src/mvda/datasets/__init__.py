"""Dataset loaders that return multi-view data as ``(views, y)``.

``views`` is a list of ``V`` arrays each shaped ``(n, d_v)``; row ``i`` of every
view is the same instance, with label ``y[i]``.
"""

from .multiple_features import load_multiple_features, train_test_split_per_class
from .colorferet import load_colorferet

__all__ = [
    "load_multiple_features",
    "train_test_split_per_class",
    "load_colorferet",
]
