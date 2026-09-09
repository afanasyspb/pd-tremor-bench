"""
Effect of class re-weighting on the feature-based models under LOSO.

Companion to `cnn_baseline.py --class-weight`: logistic regression, random
forest and the EBM are refit with inverse-frequency class weights on the same
folds, to show how re-weighting moves the operating point (balanced accuracy,
raw accuracy) without changing the ranking quality (ROC-AUC).

    python class_weight_check.py
"""
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from benchmark_loso import OUT, SEED, bootstrap_ci, load_data, safe_auc


def models():
    return {
        "Logistic Regression (balanced)": LogisticRegression(max_iter=1000, random_state=SEED,
                                                             class_weight="balanced"),
        "Random Forest (balanced)": RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                                           random_state=SEED, class_weight="balanced"),
        "EBM (Glassbox, balanced)": ExplainableBoostingClassifier(random_state=SEED, n_jobs=-1),
    }


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    X, y, groups, feats = load_data()
    rows = []
    for f_idx, (tr, te) in enumerate(LeaveOneGroupOut().split(X, y, groups=groups)):
        subj = groups[te][0]
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        ytr, yte = y[tr], y[te]
        single = len(np.unique(yte)) < 2
        w = np.where(ytr == 1, (ytr == 0).sum() / (ytr == 1).sum(), 1.0)
        for name, model in models().items():
            if name.startswith("EBM"):
                model.fit(Xtr, ytr, sample_weight=w)
            else:
                model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            proba = model.predict_proba(Xte)[:, 1]
            rows.append(dict(Model=name, Subject=subj, single_class=single,
                             Accuracy=accuracy_score(yte, pred),
                             BalAcc=balanced_accuracy_score(yte, pred) if not single else np.nan,
                             AUC=safe_auc(yte, proba),
                             F1=f1_score(yte, pred, average="weighted")))
        print(f"  fold {f_idx + 1:2d}/15 subj={subj}", flush=True)
    folds = pd.DataFrame(rows)
    folds.to_csv(OUT / "class_weight_loso_folds.csv", index=False)
    recs = []
    for m in folds.Model.unique():
        g = folds[folds.Model == m]
        two = g[~g.single_class]
        am, alo, ahi = bootstrap_ci(two.AUC.values)
        bm, blo, bhi = bootstrap_ci(two.BalAcc.values)
        fm, flo, fhi = bootstrap_ci(g.F1.values)
        recs.append(dict(Model=m, AUC_mean=am, AUC_lo=alo, AUC_hi=ahi,
                         BalAcc_mean=bm, BalAcc_lo=blo, BalAcc_hi=bhi,
                         F1_mean=fm, F1_lo=flo, F1_hi=fhi,
                         Acc_all=g.Accuracy.mean(), Acc_2class=two.Accuracy.mean()))
    summ = pd.DataFrame(recs)
    summ.to_csv(OUT / "class_weight_loso_summary.csv", index=False)
    print(summ.round(4).to_string(index=False))
