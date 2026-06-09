# VEDA — View-Embedded Discriminant Analysis

Multi-source signal fusion pipeline that learns a shared discriminative subspace
across heterogeneous data sources. Applied to financial sector classification,
digit recognition, and cross-pose face recognition.

---

## Results

### Nifty 50 — Sector classification (price/volume only, no company identity)

| Method | Accuracy |
|---|---:|
| Random baseline | 14.3% |
| Linear fusion (VEDA) | 31.31% |
| Random Forest | **47.98%** |

![Nifty 50 sector t-SNE](results/nifty50_sectors.png)

### UCI Multiple Features — 6-view digit classification

| Method | Accuracy |
|---|---:|
| MLP / SVM / RF | 97.8 – 98.4% |
| **VEDA + Ensemble** | **98.70%** |

5-fold CV: **98.85% ± 0.52%**  ·  `python experiments/cross_validation.py --folds 5`

![t-SNE](results/tsne_comparison.png)

### ColorFERET — Cross-pose face recognition

| Poses | Subjects | Accuracy |
|---|---:|---:|
| 4 angles | 200 | **95.27%** |
| 2 angles | 993 | **90.66%** |

---

## Quickstart

```bash
pip install -r requirements.txt
python experiments/nifty50_sector.py      # Nifty 50 sector classification
python experiments/baseline_comparison.py # VEDA vs MLP / SVM / RF
python experiments/run_mvda.py --mode concat --classifier ensemble
python3 -m pytest
```

---

## Stack

Python · NumPy · SciPy · scikit-learn · yfinance · Matplotlib
