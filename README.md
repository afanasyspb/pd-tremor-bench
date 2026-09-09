# Patient-independent tremor-state classification on TremorDB

Code for the paper

> I. Afanasyev. Towards Safety-Aware Edge AI for Parkinson's Tremor: A Feasibility Study of
> Interpretable Glassbox Ensembles and Out-of-Distribution Monitoring. EXPLAINS 2026
> (Springer CCIS).

The scripts reproduce every table and figure of the paper: a leave-one-subject-out (LOSO)
benchmark of nine feature-based classifiers and a raw-signal 1D-CNN on the public PhysioNet
TremorDB recordings (rest tremor of 15 subjects with deep brain stimulation ON or OFF),
subject-level statistics, an Isolation Forest out-of-distribution monitor, a per-subject
normalization experiment, and a subject-blind random-split comparison.

## Setup

Python 3.12.

```bash
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate
pip install -r requirements.txt
```

`torch` is only needed for `cnn_baseline.py`; the CPU build is enough.

## Data

`data/tremordb_window_features.csv` is included: 6627 windows of 2.0 s (1.0 s hop) with
10 spectro-temporal features and metadata. The raw recordings are only needed for the CNN
baseline or to rebuild the table; see `data/README.md` for the download command.

## Scripts

| Script | What it does | Output (`results/`) |
|---|---|---|
| `extract_features.py` | Builds the feature table from the raw recordings | `data/tremordb_window_features.csv` |
| `benchmark_loso.py` | LOSO benchmark of the nine feature-based models, OOD monitor, bootstrap CIs, Wilcoxon tests, per-subject z-score EBM, EBM term importances | `summary_corrected.csv`, `folds_corrected.csv`, `stats_wilcoxon.csv`, `ood_per_fold.csv`, `personalized_ebm_*.csv`, `ebm_native_importance.csv` |
| `random_split.py` | Same nine models under a subject-blind 5-fold window split | `random_split_window_level_*.csv` |
| `cnn_baseline.py` | 1D-CNN on the raw 2 s signal, LOSO and random split (`--class-weight` for the weighted loss) | `cnn_loso_*.csv`, `cnn_random_split_*.csv` |
| `class_weight_check.py` | Logistic regression, random forest and EBM refit with inverse-frequency class weights on the same LOSO folds | `class_weight_loso_*.csv` |
| `make_figures.py` | Draws the figures from the CSV files | `fig3_*.pdf`, `fig4_*.pdf`, `fig5_*.pdf` |
| `scn.py` | Stochastic Configuration Network used by the benchmark | |

Run them in this order:

```bash
python benchmark_loso.py      # about 10 minutes on a laptop CPU
python random_split.py
python cnn_baseline.py        # needs data/tremordb, about 10 minutes on CPU
python cnn_baseline.py --class-weight
python class_weight_check.py
python make_figures.py
```

`results/` contains the files produced for the paper, so the numbers can be checked without
re-running anything. All random seeds are fixed (`SEED = 42`); the benchmark has been verified
to reproduce the committed CSV files exactly with the pinned package versions.

## Metrics

Windows are labelled 1 for DBS OFF and 0 for DBS ON. ROC-AUC and balanced accuracy are
averaged over the 13 subjects that have both classes; accuracy, F1, the noise stress test
and latency use all 15 subjects. Confidence intervals are subject-level bootstraps
(10,000 resamples), and the Wilcoxon tests are paired over subjects.

## License and citation

The code is released under the MIT License (see `LICENSE`).

Dataset: Beuter A, Titcombe MS, Richer F, Gross C, Guehl D. Effect of deep brain stimulation on
amplitude and frequency characteristics of rest tremor in Parkinson's disease. Thalamus &
Related Systems 1(3):203-211, 2001. Distributed through PhysioNet
(https://physionet.org/content/tremordb/1.0.0/).
