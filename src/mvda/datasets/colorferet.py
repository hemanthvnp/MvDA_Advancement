"""ColorFERET multi-view face loader.

ColorFERET stores one or more images per subject under different *poses*
(frontal, quarter-left, profile, ...). We treat each pose as a *view* and each
subject as a *class*, giving a genuine multi-view problem: the same person seen
from several angles, with instance correspondence by subject.

The loader is **path-agnostic**: point ``root`` at any directory that contains
the image files -- a local copy, an ``rclone`` mount, or a Google-Drive mount in
Colab (``/content/drive/MyDrive/...``). See ``docs/COLORFERET.md``.

FERET image filenames encode subject and pose, e.g. ``00123_940928_fa.ppm.bz2``
(``00123`` = subject id, ``fa`` = pose). Supported image types: ``.ppm``,
``.ppm.bz2``, ``.png``, ``.jpg``/``.jpeg`` (bz2-compressed variants too).
"""

from __future__ import annotations

import bz2
import io
import os
from collections import defaultdict
from typing import List, Optional, Sequence, Tuple

import numpy as np

# FERET pose codes.
POSE_CODES = ["pl", "pr", "hl", "hr", "ql", "qr", "ra", "rb", "rc", "rd", "re", "fa", "fb"]
_POSE_SET = set(POSE_CODES)


def _parse_name(filename: str):
    """Extract (subject, pose) from a FERET filename, or None.

    Filenames look like ``<subject>_<date>_<pose>[_<variant>].<ext>[.bz2]``,
    e.g. ``00001_930831_fa.ppm.bz2`` or ``00001_930831_fa_a.ppm.bz2`` (the
    ``_a``/``_b`` variants must not be skipped). We split on the extension and
    on ``_`` and take the first token that is a known pose code, which is
    robust to such trailing variant suffixes.
    """
    stem = filename.split(".")[0]
    parts = stem.split("_")
    if not parts or len(parts[0]) != 5 or not parts[0].isdigit():
        return None
    for tok in parts[1:]:
        if tok in _POSE_SET:
            return parts[0], tok
    return None
_IMAGE_EXT = (".ppm", ".png", ".jpg", ".jpeg")


def _is_image(name: str) -> bool:
    low = name.lower()
    if low.endswith(".bz2"):
        low = low[:-4]
    return low.endswith(_IMAGE_EXT)


def _read_image(path: str, size: Tuple[int, int], grayscale: bool) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise ImportError(
            "Reading ColorFERET images requires Pillow. Install it with "
            "`pip install pillow`."
        ) from exc

    if path.lower().endswith(".bz2"):
        with bz2.open(path, "rb") as f:
            raw = f.read()
        img = Image.open(io.BytesIO(raw))
    else:
        img = Image.open(path)

    img = img.convert("L" if grayscale else "RGB").resize(size)
    return np.asarray(img, dtype=np.float64).reshape(-1)


def _scan(root: str):
    """Return {subject: {pose: filepath}} by recursively scanning ``root``."""
    table = defaultdict(dict)
    for dirpath, _, files in os.walk(root):
        for name in files:
            if not _is_image(name):
                continue
            parsed = _parse_name(name)
            if not parsed:
                continue
            subject, pose = parsed
            # First image wins if a subject has duplicates of the same pose.
            table[subject].setdefault(pose, os.path.join(dirpath, name))
    return table


def load_colorferet(
    root: str,
    poses: Sequence[str] = ("ql", "fa", "qr"),
    image_size: Tuple[int, int] = (64, 64),
    grayscale: bool = True,
    max_subjects: Optional[int] = None,
    cache_path: Optional[str] = None,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Build multi-view face data from a ColorFERET image tree.

    Parameters
    ----------
    root : directory containing the FERET image files (scanned recursively).
    poses : the pose codes to use as views; subjects missing any are dropped.
    image_size : (width, height) each image is resized to before flattening.
    grayscale : load as grayscale (1 channel) vs RGB (3 channels).
    max_subjects : optionally cap the number of classes (useful for quick runs).
    cache_path : if given, assembled arrays are cached to/loaded from this .npz.

    Returns ``(views, y)`` with one view per pose, aligned by subject.
    """
    poses = list(poses)
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return [data[f"view_{i}"] for i in range(len(poses))], data["y"]

    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"ColorFERET root not found: {root!r}. Point it at a local copy or a "
            f"mounted Drive path (see docs/COLORFERET.md)."
        )

    table = _scan(root)
    subjects = sorted(s for s, p in table.items() if all(pose in p for pose in poses))
    if not subjects:
        raise RuntimeError(
            f"No subjects have all requested poses {poses}. Found {len(table)} "
            f"subjects total under {root}. Try fewer/different poses."
        )
    if max_subjects:
        subjects = subjects[:max_subjects]

    views = [np.zeros((len(subjects), image_size[0] * image_size[1] * (1 if grayscale else 3)))
             for _ in poses]
    for row, subject in enumerate(subjects):
        for vi, pose in enumerate(poses):
            views[vi][row] = _read_image(table[subject][pose], image_size, grayscale)

    y = np.arange(len(subjects))  # encoded subject id -> class label

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez_compressed(cache_path, y=y, **{f"view_{i}": v for i, v in enumerate(views)})

    return views, y
