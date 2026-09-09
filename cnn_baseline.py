"""
1D-CNN baseline on the raw 2 s velocity windows.

The network uses the same 6627 windows as the feature-based models but reads
the raw 200-sample signal instead of the 10 engineered features. It is
evaluated under both protocols: strict leave-one-subject-out (with the
Gaussian-noise stress test) and the subject-blind 5-fold window split of
random_split.py. Each window is mean-centred, as in feature extraction, and
scaled by one global standard deviation estimated on the training windows of
the fold, so the amplitude differences between windows are preserved.

Requires the raw PhysioNet recordings in data/tremordb (see data/README.md).

    python cnn_baseline.py                 # plain binary cross-entropy
    python cnn_baseline.py --class-weight  # positive class weighted by n_neg/n_pos
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from benchmark_loso import DATA_PATH, NOISE_SIGMA, OUT, ROOT, SEED, bootstrap_ci, safe_auc
from extract_features import parse_file_description, read_signal_txt

RAW_DIR = ROOT / "data" / "tremordb"
FS, WIN = 100, 200
EPOCHS, BATCH, LR, WD = 30, 128, 1e-3, 1e-4
MODEL_NAME = "1D-CNN (raw signal)"
CLASS_WEIGHT = False
SUFFIX = ""


def load_raw_windows():
    """Raw 200-sample window behind every row of the feature table, in mm/s."""
    df = pd.read_csv(DATA_PATH).dropna(subset=["target_label"])
    meta = parse_file_description(RAW_DIR / "file_description.txt")
    signals = {}
    X = np.empty((len(df), WIN), dtype=np.float32)
    for i, (rec, start) in enumerate(zip(df.record, df.window_start_s)):
        if rec not in signals:
            path = next(RAW_DIR.rglob(rec), None)
            if path is None:
                raise FileNotFoundError(f"{rec} not found under {RAW_DIR}")
            x = read_signal_txt(path)
            if meta[rec]["unit"] == "m/s":
                x = x * 1000.0
            signals[rec] = x
        s = int(round(start * FS))
        w = signals[rec][s:s + WIN]
        if len(w) != WIN:
            raise ValueError(f"{rec} @ {start}s: {len(w)} samples")
        X[i] = w
    y = df["target_label"].values.astype(int)
    groups = df["subject"].values
    print(f"Raw windows: {X.shape} | subjects: {len(np.unique(groups))}")
    return X, y, groups


class TremorCNN(nn.Module):
    """Two conv blocks, global average pooling, small dense head (about 49k parameters)."""

    def __init__(self, c1=48, c2=96, k=9, hidden=64, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, c1, k, padding=k // 2), nn.BatchNorm1d(c1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, k, padding=k // 2), nn.BatchNorm1d(c2), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(c2, hidden), nn.ReLU(), nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def standardize(Xtr, Xte):
    Xtr = Xtr - Xtr.mean(axis=1, keepdims=True)
    Xte = Xte - Xte.mean(axis=1, keepdims=True)
    sd = Xtr.std()
    return (Xtr / sd).astype(np.float32), (Xte / sd).astype(np.float32)


def train_model(Xtr, ytr, seed):
    torch.manual_seed(seed)
    model = TremorCNN()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    if CLASS_WEIGHT:
        pos_weight = torch.tensor((ytr == 0).sum() / max(1, (ytr == 1).sum()), dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(Xtr).unsqueeze(1)
    yt = torch.from_numpy(ytr.astype(np.float32))
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(Xt), generator=gen)
        for b in range(0, len(Xt), BATCH):
            idx = perm[b:b + BATCH]
            opt.zero_grad()
            loss_fn(model(Xt[idx]), yt[idx]).backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model, X):
    proba = torch.sigmoid(model(torch.from_numpy(X).unsqueeze(1))).numpy()
    return (proba >= 0.5).astype(int), proba


def evaluate(model, Xte, yte, single_class):
    t0 = time.perf_counter()
    pred, proba = predict(model, Xte)
    infer_ms = (time.perf_counter() - t0) * 1e3 / len(yte)
    return dict(
        Accuracy=accuracy_score(yte, pred),
        BalAcc=balanced_accuracy_score(yte, pred) if not single_class else np.nan,
        AUC=safe_auc(yte, proba),
        F1=f1_score(yte, pred, average="weighted"),
        Infer_ms_per_sample=infer_ms,
    )


def run_loso(X, y, groups):
    rows = []
    for f_idx, (tr, te) in enumerate(LeaveOneGroupOut().split(X, y, groups=groups)):
        subj = groups[te][0]
        rng = np.random.default_rng(SEED + f_idx)
        Xtr, Xte = standardize(X[tr], X[te])
        ytr, yte = y[tr], y[te]
        single = len(np.unique(yte)) < 2
        t0 = time.perf_counter()
        model = train_model(Xtr, ytr, SEED + f_idx)
        train_s = time.perf_counter() - t0
        m = evaluate(model, Xte, yte, single)
        Xte_noisy = (Xte + rng.normal(0, NOISE_SIGMA, Xte.shape)).astype(np.float32)
        mn = evaluate(model, Xte_noisy, yte, single)
        rows.append(dict(Model=MODEL_NAME, Subject=subj, single_class=single, **m,
                         Acc_drop=m["Accuracy"] - mn["Accuracy"],
                         F1_drop=m["F1"] - mn["F1"],
                         AUC_drop=(m["AUC"] - mn["AUC"]) if not np.isnan(m["AUC"]) else np.nan,
                         Train_s=train_s))
        print(f"  fold {f_idx + 1:2d}/15 subj={subj:4s} acc={m['Accuracy']:.3f} "
              f"bacc={m['BalAcc']:.3f} auc={m['AUC']:.3f} f1={m['F1']:.3f} "
              f"dF1={rows[-1]['F1_drop']:+.3f} ({train_s:.0f}s)"
              f"{'  [single-class]' if single else ''}", flush=True)
    return pd.DataFrame(rows)


def summarise_loso(folds):
    two = folds[~folds.single_class]
    am, alo, ahi = bootstrap_ci(two.AUC.values)
    bm, blo, bhi = bootstrap_ci(two.BalAcc.values)
    fm, flo, fhi = bootstrap_ci(two.F1.values)
    return pd.DataFrame([dict(
        Model=MODEL_NAME, n_AUC=len(two),
        AUC_mean=am, AUC_lo=alo, AUC_hi=ahi,
        F1_mean=fm, F1_lo=flo, F1_hi=fhi,
        BalAcc_mean=bm, BalAcc_lo=blo, BalAcc_hi=bhi,
        Acc_all=folds.Accuracy.mean(), Acc_2class=two.Accuracy.mean(),
        F1_drop_noise=folds.F1_drop.mean(),
        AUC_drop_noise=folds.AUC_drop.dropna().mean(),
        Acc_drop_noise=folds.Acc_drop.mean(),
        Infer_ms=folds.Infer_ms_per_sample.mean())])


def compare_with_ebm(folds):
    ref_path = OUT / "folds_corrected.csv"
    if not ref_path.exists():
        return
    ref = pd.read_csv(ref_path)
    ebm = ref[ref.Model == "EBM (Glassbox)"].set_index("Subject")
    cnn = folds.set_index("Subject")
    common = ebm.AUC.dropna().index.intersection(cnn.AUC.dropna().index)
    for metric in ("AUC", "BalAcc"):
        a, b = cnn.loc[common, metric].values, ebm.loc[common, metric].values
        p = stats.wilcoxon(a, b, alternative="two-sided").pvalue
        print(f"  CNN vs EBM {metric}: mean {a.mean():.4f} vs {b.mean():.4f}, "
              f"CNN higher on {(a > b).sum()}/{len(common)} subjects, Wilcoxon p={p:.4f}")


def run_random_split(X, y):
    rows = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for k, (tr, te) in enumerate(skf.split(X, y)):
        Xtr, Xte = standardize(X[tr], X[te])
        model = train_model(Xtr, y[tr], SEED + k)
        m = evaluate(model, Xte, y[te], False)
        rows.append(dict(Model=MODEL_NAME, fold=k, Accuracy=m["Accuracy"],
                         BalAcc=m["BalAcc"], AUC=m["AUC"]))
        print(f"  random-split fold {k} acc={m['Accuracy']:.4f} bacc={m['BalAcc']:.4f} "
              f"auc={m['AUC']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    summ = df.groupby("Model")[["Accuracy", "BalAcc", "AUC"]].agg(["mean", "std"])
    summ.columns = ["_".join(c) for c in summ.columns]
    return df, summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-weight", action="store_true",
                    help="weight the positive class by n_neg/n_pos in the loss")
    args = ap.parse_args()
    if args.class_weight:
        CLASS_WEIGHT, SUFFIX = True, "_weighted"
        MODEL_NAME = "1D-CNN (raw signal, class-weighted)"

    OUT.mkdir(exist_ok=True)
    print(f"torch {torch.__version__}, threads {torch.get_num_threads()}, "
          f"parameters {n_params(TremorCNN()):,}, class_weight={CLASS_WEIGHT}")
    X, y, groups = load_raw_windows()

    print("\nLOSO ...")
    folds = run_loso(X, y, groups)
    folds.to_csv(OUT / f"cnn_loso_folds{SUFFIX}.csv", index=False)
    summ = summarise_loso(folds)
    summ.to_csv(OUT / f"cnn_loso_summary{SUFFIX}.csv", index=False)
    print("\nLOSO summary:")
    print(summ.round(4).T.to_string())
    compare_with_ebm(folds)

    print("\nWindow-level random split ...")
    rs_folds, rs_summ = run_random_split(X, y)
    rs_folds.to_csv(OUT / f"cnn_random_split_folds{SUFFIX}.csv", index=False)
    rs_summ.to_csv(OUT / f"cnn_random_split_summary{SUFFIX}.csv")
    print(rs_summ.round(4).to_string())
