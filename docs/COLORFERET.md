# Using the ColorFERET dataset

ColorFERET is a licensed face dataset (subjects photographed under several
*poses*). This project treats **each pose as a view** and **each subject as a
class**, which makes it a natural multi-view benchmark for MvDA.

The loader (`mvda.datasets.load_colorferet`) is **path-agnostic**: it
recursively scans whatever directory you give it for FERET image files, parses
the subject id and pose from each filename (e.g. `00123_940928_fa.ppm.bz2`),
and assembles one view per requested pose, aligned by subject.

> **Note on the bundled `colorferet/` folder:** the copy committed to early
> versions of this repo contained only the *ground-truth metadata*
> (`name_value/*.txt`, `xml/*.xml`) — **no images**. You must supply the image
> files yourself (they are distributed under license by NIST).

## Option A — Google Colab (mount Drive, no manual download)

If your images live in Google Drive, run on Colab and mount the drive so the
folder appears as a normal filesystem path:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then point the loader at the mounted path:

```bash
python experiments/run_mvda.py --dataset colorferet --mode mvda \
    --feret-root /content/drive/MyDrive/colorferet \
    --feret-poses ql fa qr
```

## Option B — Local copy / rclone / Google Drive for Desktop

Mount Drive as a drive letter (Google Drive for Desktop or `rclone mount`), or
copy the images locally, then:

```bash
python experiments/run_mvda.py --dataset colorferet --mode mvda \
    --feret-root "G:/My Drive/colorferet"
```

## Option C — Download the Drive folder with gdown

```bash
pip install gdown
python -m gdown --folder "<your-drive-folder-url>" -O data/feret_raw
```

The Drive folder must be shared as **"Anyone with the link."** `gdown` enumerates
the whole tree before downloading, so large folders take a while.

## Choosing poses

`--feret-poses` selects which poses become views. Subjects missing **any** of
the requested poses are dropped (multi-view requires correspondence), so more
poses ⇒ fewer usable subjects. Sensible starting points:

| poses | meaning | trade-off |
|-------|---------|-----------|
| `ql fa qr` (default) | quarter-left, frontal, quarter-right | good coverage, 3 views |
| `fa fb` | two frontal captures | most subjects, easiest |
| `pl hl ql fa qr hr pr` | full pose sweep | richest, fewest subjects |

Tune image size / subject cap for quick runs:

```bash
python experiments/run_mvda.py --dataset colorferet \
    --feret-size 48 48 --feret-max-subjects 100
```

Assembled views are cached to `data/feret_cache.npz` so repeat runs are fast.
