"""
Subject-blind window-level split for the nine feature-based models.

Same features and model configurations as benchmark_loso.py, but evaluated
with a stratified, shuffled 5-fold split over windows, so every subject (and
the overlapping neighbours of every window) appears in both training and test
sets. Used to quantify the inflation caused by this protocol.

    python random_split.py
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from benchmark_loso import OUT, SEED, build_models, load_data

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    np.random.seed(SEED)
    X, y, groups, feats = load_data()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for k, (tr, te) in enumerate(skf.split(X, y)):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        for name, model in build_models().items():
            t0 = time.time()
            model.fit(Xtr, y[tr])
            pred = model.predict(Xte)
            try:
                auc = roc_auc_score(y[te], model.predict_proba(Xte)[:, 1])
            except Exception:
                auc = np.nan
            rows.append(dict(Model=name, fold=k,
                             Accuracy=accuracy_score(y[te], pred),
                             BalAcc=balanced_accuracy_score(y[te], pred),
                             AUC=auc))
            print(f"fold {k} {name:28s} acc={rows[-1]['Accuracy']:.4f} "
                  f"bacc={rows[-1]['BalAcc']:.4f} auc={auc:.4f} ({time.time() - t0:.1f}s)",
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "random_split_window_level_folds.csv", index=False)
    summ = df.groupby("Model")[["Accuracy", "BalAcc", "AUC"]].agg(["mean", "std"])
    summ.columns = ["_".join(c) for c in summ.columns]
    summ.to_csv(OUT / "random_split_window_level_summary.csv")
    print("\nWindow-level 5-fold split (stratified, shuffled, subject-blind):")
    print(summ.round(4).to_string())
