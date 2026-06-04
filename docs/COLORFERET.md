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
python experiments/run_feret.py \
    --feret-root /content/drive/MyDrive/colorferet \
    --feret-poses fa fb hl hr --pca 120
```

## Option B — Local copy / rclone / Google Drive for Desktop

Mount Drive as a drive letter (Google Drive for Desktop or `rclone mount`), or
copy the images locally, then:

```bash
python experiments/run_feret.py --feret-root "G:/My Drive/colorferet"
```

## Option C — Download the Drive folder with gdown

```bash
pip install gdown
python -m gdown --folder "<your-drive-folder-url>" -O data/feret_raw
```

The Drive folder must be shared as **"Anyone with the link."** `gdown` enumerates
the whole tree before downloading, so large folders take a while.

## Protocols

`run_feret.py` treats **pose = view** and **subject = class**; each view is
reduced with PCA (eigenfaces) before MvDA. Two evaluation protocols:

- **`--protocol disjoint` (MvDA paper, Kan et al. 2016, default).** 7 poses as
  views; the first `--train-subjects` (231) identities with `--images-per-pose`
  (4) images each train the shared subspace; the *remaining, unseen* identities
  are recognized gallery/probe — the `--gallery-pose` (fa) gives one reference
  per test subject, and every other-pose image is a probe matched by cosine
  nearest neighbour. Reports rank-1 recognition accuracy.

  ```bash
  python experiments/run_feret.py --protocol disjoint --solver ratio \
      --feret-root "$FERET_ROOT" --feret-poses pl hl ql fa qr hr pr \
      --train-subjects 231 --images-per-pose 4 --gallery-pose fa --pca 120
  ```

- **`--protocol closed`.** Per-view stratified split of all subjects; classify
  held-out single-pose images by nearest class mean.

Swap `--solver ratio|exponential|harmonic` to compare discriminant criteria
(see [FINDINGS.md](FINDINGS.md)).

## Choosing poses

`--feret-poses` selects which poses become views. A subject must appear in
**every** requested pose (so all classes exist in all views), so more poses ⇒
fewer usable subjects. Sensible starting points:

| poses | meaning | trade-off |
|-------|---------|-----------|
| `fa fb hl hr` (default) | frontal x2, half-left, half-right | good balance |
| `fa fb` | two frontal captures | most subjects, easiest |
| `pl hl ql fa qr hr pr` | full pose sweep | richest, fewest subjects |

Tune image size / PCA dims / subject cap for quick runs:

```bash
python experiments/run_feret.py --feret-root "$FERET_ROOT" \
    --feret-size 48 48 --pca 80 --feret-max-subjects 100
```

Assembled views are cached under `data/feret_<config>.npz` (keyed to the
poses/size) so repeat runs are fast; pass `--no-cache` to disable.
