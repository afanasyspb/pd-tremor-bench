"""
Leave-one-subject-out benchmark of nine feature-based classifiers.

For every held-out subject the script fits a StandardScaler on the training
subjects, trains the nine models, and records accuracy, balanced accuracy,
ROC-AUC, weighted F1, the metric drop under Gaussian input noise and the
inference latency. One Isolation Forest per fold serves as a model-agnostic
out-of-distribution monitor. Subject-level bootstrap confidence intervals,
paired Wilcoxon tests with Holm correction, a per-subject z-score variant of
the EBM and the native EBM term importances are written to results/.

    python benchmark_loso.py
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (GradientBoostingClassifier, IsolationForest,
                              RandomForestClassifier)
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
from scn import StochasticConfigurationNetwork

SEED = 42
NOISE_SIGMA = 0.5       # std of the additive noise in standardized feature space
OOD_SIGMA = 3.0         # std of the synthetic out-of-distribution samples
BOOT = 10000            # bootstrap resamples
DATA_PATH = ROOT / "data" / "tremordb_window_features.csv"
OUT = ROOT / "results"

META_COLS = ["record", "subject", "hand", "condition", "dbs_state", "med_state",
             "mins_after_dbs_stop", "tremor_group", "fs_hz", "unit",
             "target_label", "window_start_s", "window_size_s"]

FEATURE_LABELS = {
    "rms": "RMS", "std": "Std. Deviation", "p2p": "Peak-to-Peak", "mad": "MAD",
    "peak_freq_hz": "Peak Frequency (Hz)", "peak_psd": "Peak PSD",
    "bp_3_12": "Band Power 3-12 Hz", "bp_4_6": "Band Power 4-6 Hz",
    "bp_6_12": "Band Power 6-12 Hz", "rel_bp_3_12": "Rel. Band Power 3-12 Hz",
}


def load_data():
    df = pd.read_csv(DATA_PATH).dropna(subset=["target_label"])
    feats = [c for c in df.columns if c not in META_COLS]
    X = df[feats].values.astype(float)
    y = df["target_label"].values.astype(int)
    groups = df["subject"].values
    print(f"Features ({len(feats)}): {feats}")
    print(f"Windows: {X.shape[0]} | Subjects: {len(np.unique(groups))}")
    for s in np.unique(groups):
        if len(np.unique(y[groups == s])) < 2:
            print(f"  subject {s}: single class, excluded from ROC-AUC and balanced accuracy")
    return X, y, groups, feats


class SCNClassifier:
    """scikit-learn style wrapper around the SCN ensemble."""

    def __init__(self, num_hidden=150, num_runs=30, random_state=SEED):
        self.model = StochasticConfigurationNetwork(
            num_hidden=num_hidden, num_runs=num_runs,
            random_state=random_state, verbose=False)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        mean_proba, _ = self.model.predict_proba(X)
        return np.column_stack((1.0 - mean_proba, mean_proba))


def build_models():
    """Fresh instances of the nine models (called once per fold)."""
    rff_ridge = Pipeline([
        ("rff", RBFSampler(n_components=500, gamma="scale", random_state=SEED)),
        ("ridge", CalibratedClassifierCV(RidgeClassifier(alpha=0.001),
                                         method="sigmoid", cv=3))])
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=SEED),
        "CatBoost": CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05,
                                       verbose=0, random_seed=SEED),
        "EBM (Glassbox)": ExplainableBoostingClassifier(random_state=SEED, n_jobs=-1),
        "SVM (RBF Kernel)": SVC(kernel="rbf", probability=True, random_state=SEED),
        "MLP Neural Network": MLPClassifier(hidden_layer_sizes=(100, 50),
                                            max_iter=500, random_state=SEED),
        "RFF + Ridge (Proposed)": rff_ridge,
        "SCN (Stochastic Config)": SCNClassifier(150, 30, SEED),
    }


def fold_ood_auroc(Xtr, Xte, rng):
    """AUROC of an Isolation Forest at separating the held-out windows from
    synthetic Gaussian samples of the same shape."""
    iso = IsolationForest(contamination=0.05, random_state=SEED).fit(Xtr)
    s_in = iso.decision_function(Xte)
    s_out = iso.decision_function(rng.normal(0, OOD_SIGMA, Xte.shape))
    y_true = np.r_[np.ones_like(s_in), np.zeros_like(s_out)]
    return roc_auc_score(y_true, np.r_[s_in, s_out])


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def run_loso(X, y, groups):
    """Outer loop over held-out subjects, inner loop over models. The OOD monitor
    and the noise realisation are drawn once per fold and shared by all models."""
    logo = LeaveOneGroupOut()
    rows, ood_rows = [], []
    for f_idx, (tr, te) in enumerate(logo.split(X, y, groups=groups)):
        subj = groups[te][0]
        rng = np.random.default_rng(SEED + f_idx)
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        ytr, yte = y[tr], y[te]
        single_class = len(np.unique(yte)) < 2

        ood = fold_ood_auroc(Xtr, Xte, rng)
        ood_rows.append({"Subject": subj, "OOD_AUROC": ood, "single_class": single_class})
        Xte_noisy = Xte + rng.normal(0, NOISE_SIGMA, Xte.shape)

        for name, model in build_models().items():
            t0 = time.perf_counter()
            model.fit(Xtr, ytr)
            train_ms = (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter()
            y_pred = model.predict(Xte)
            infer_ms = (time.perf_counter() - t0) * 1e3 / len(yte)
            try:
                proba = model.predict_proba(Xte)[:, 1]
            except Exception:
                proba = None
            acc = accuracy_score(yte, y_pred)
            bal_acc = balanced_accuracy_score(yte, y_pred) if not single_class else np.nan
            f1 = f1_score(yte, y_pred, average="weighted")
            auc = safe_auc(yte, proba) if proba is not None else np.nan

            y_pred_n = model.predict(Xte_noisy)
            try:
                proba_n = model.predict_proba(Xte_noisy)[:, 1]
            except Exception:
                proba_n = None
            acc_n = accuracy_score(yte, y_pred_n)
            f1_n = f1_score(yte, y_pred_n, average="weighted")
            auc_n = safe_auc(yte, proba_n) if proba_n is not None else np.nan

            rows.append({
                "Model": name, "Subject": subj, "single_class": single_class,
                "Accuracy": acc, "BalAcc": bal_acc, "AUC": auc, "F1": f1,
                "Acc_drop": acc - acc_n,
                "F1_drop": f1 - f1_n,
                "AUC_drop": (auc - auc_n) if not np.isnan(auc) else np.nan,
                "Infer_ms_per_sample": infer_ms, "Train_ms": train_ms})
        print(f"  fold {f_idx + 1:2d}/{len(np.unique(groups))}  subj={subj:5s}"
              f"  OOD={ood:.3f}{'  [single-class]' if single_class else ''}")
    return pd.DataFrame(rows), pd.DataFrame(ood_rows)


def bootstrap_ci(values, B=BOOT, seed=SEED, alpha=0.05):
    v = np.asarray([x for x in values if not np.isnan(x)], float)
    if v.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(B, v.size))].mean(axis=1)
    return v.mean(), np.percentile(means, 100 * alpha / 2), np.percentile(means, 100 * (1 - alpha / 2))


def wilcoxon_pair(df, metric, a, b):
    """Exact paired Wilcoxon signed-rank test over the folds where both models
    have a defined metric."""
    pa = df[df.Model == a].set_index("Subject")[metric]
    pb = df[df.Model == b].set_index("Subject")[metric]
    common = pa.dropna().index.intersection(pb.dropna().index)
    x, y = pa.loc[common].values, pb.loc[common].values
    d = x - y
    if np.allclose(d, 0):
        return dict(n=len(common), median_d=0.0, mean_d=0.0, p=1.0)
    p = None
    for kw in ({"method": "exact"}, {"mode": "exact"}, {}):
        try:
            p = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", **kw).pvalue
            break
        except TypeError:
            continue
    if p is None:
        p = stats.wilcoxon(x, y, alternative="two-sided").pvalue
    return dict(n=len(common), median_d=float(np.median(d)), mean_d=float(np.mean(d)), p=float(p))


def holm(labels_pvals):
    order = sorted(labels_pvals, key=lambda t: t[1])
    m, out, run = len(order), {}, 0.0
    for k, (lab, p) in enumerate(order):
        run = max(run, min(1.0, (m - k) * p))
        out[lab] = run
    return out


def summarise(folds, ood):
    models = list(dict.fromkeys(folds.Model))
    recs = []
    for m in models:
        g = folds[folds.Model == m]
        aucs = g.AUC.dropna().values
        f1s = g.F1.values
        baccs = g.BalAcc.dropna().values
        am, alo, ahi = bootstrap_ci(aucs)
        fm, flo, fhi = bootstrap_ci(f1s)
        bam, balo, bahi = bootstrap_ci(baccs)
        recs.append({
            "Model": m, "n_AUC": len(aucs),
            "AUC_mean": am, "AUC_lo": alo, "AUC_hi": ahi,
            "F1_mean": fm, "F1_lo": flo, "F1_hi": fhi,
            "BalAcc_mean": bam, "BalAcc_lo": balo, "BalAcc_hi": bahi,
            "Acc_all": g.Accuracy.values.mean(),
            "Acc_2class": g[~g.single_class].Accuracy.values.mean(),
            "F1_drop_noise": g.F1_drop.mean(),
            "AUC_drop_noise": g.AUC_drop.dropna().mean(),
            "Acc_drop_noise": g.Acc_drop.mean(),
            "Infer_ms": g.Infer_ms_per_sample.mean()})
    summ = pd.DataFrame(recs)
    om, olo, ohi = bootstrap_ci(ood.OOD_AUROC.values)
    print("\nSUMMARY (95% bootstrap CI over subjects)")
    print(summ.round(4).to_string(index=False))
    print(f"\nOOD monitor (Isolation Forest vs Gaussian samples): mean={om:.4f} "
          f"95%CI=[{olo:.4f},{ohi:.4f}] min={ood.OOD_AUROC.min():.4f} "
          f"(subject {ood.loc[ood.OOD_AUROC.idxmin(), 'Subject']})")
    summ.to_csv(OUT / "summary_corrected.csv", index=False)
    ood.to_csv(OUT / "ood_per_fold.csv", index=False)
    return summ


def run_stats(folds):
    pairs = [("SCN (Stochastic Config)", "EBM (Glassbox)"),
             ("CatBoost", "EBM (Glassbox)"),
             ("SCN (Stochastic Config)", "CatBoost"),
             ("EBM (Glassbox)", "Logistic Regression"),
             ("SCN (Stochastic Config)", "Logistic Regression")]
    out = []
    for metric in ("AUC", "F1"):
        raw = {f"{a}|{b}": wilcoxon_pair(folds, metric, a, b) for a, b in pairs}
        hp = holm([(k, v["p"]) for k, v in raw.items()])
        print(f"\nWilcoxon signed-rank ({metric}), Holm-corrected")
        print(f"{'pair':26}{'n':>3}{'med_d':>9}{'mean_d':>9}{'p_raw':>9}{'p_holm':>9}")
        for k, v in raw.items():
            a, b = k.split("|")
            lab = f"{a.split()[0]} vs {b.split()[0]}"
            print(f"{lab:26}{v['n']:>3}{v['median_d']:>9.4f}{v['mean_d']:>9.4f}"
                  f"{v['p']:>9.4f}{hp[k]:>9.4f}")
            out.append(dict(metric=metric, pair=lab, **v, p_holm=hp[k]))
    pd.DataFrame(out).to_csv(OUT / "stats_wilcoxon.csv", index=False)


def per_subject_zscore(X, groups):
    """Standardize every subject with its own mean and std (label-free)."""
    Xp = np.empty_like(X, dtype=float)
    for s in np.unique(groups):
        m = groups == s
        Xp[m] = StandardScaler().fit_transform(X[m])
    return Xp


def loso_ebm(X, y, groups):
    rows = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups=groups):
        subj = groups[te][0]
        ytr, yte = y[tr], y[te]
        single = len(np.unique(yte)) < 2
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        ebm = ExplainableBoostingClassifier(random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
        y_pred = ebm.predict(Xte)
        proba = ebm.predict_proba(Xte)[:, 1]
        rows.append({"Subject": subj, "single_class": single,
                     "Accuracy": accuracy_score(yte, y_pred),
                     "BalAcc": balanced_accuracy_score(yte, y_pred) if not single else np.nan,
                     "AUC": safe_auc(yte, proba),
                     "F1": f1_score(yte, y_pred, average="weighted")})
    return pd.DataFrame(rows)


def run_personalized_ebm(X, y, groups):
    """EBM with global train-fold scaling versus EBM with per-subject z-scoring,
    on identical LOSO folds."""
    print("\nEBM: global train-fold scaling vs per-subject z-score")
    agn = loso_ebm(X, y, groups)
    per = loso_ebm(per_subject_zscore(X, groups), y, groups)
    agn.to_csv(OUT / "personalized_ebm_agnostic_folds.csv", index=False)
    per.to_csv(OUT / "personalized_ebm_folds.csv", index=False)

    def summ(df, tag):
        am, alo, ahi = bootstrap_ci(df.AUC.dropna().values)
        bm, blo, bhi = bootstrap_ci(df.BalAcc.dropna().values)
        fm, flo, fhi = bootstrap_ci(df.F1.values)
        print(f"  {tag}")
        print(f"    ROC-AUC {am:.4f} [{alo:.4f}, {ahi:.4f}]  Bal.Acc {bm:.4f} [{blo:.4f}, {bhi:.4f}]"
              f"  F1 {fm:.4f} [{flo:.4f}, {fhi:.4f}]  Acc(13) {df[~df.single_class].Accuracy.mean():.4f}")
        return am, bm, fm

    a_auc, a_bacc, a_f1 = summ(agn, "global scaling:")
    p_auc, p_bacc, p_f1 = summ(per, "per-subject z-score:")

    ai, pi = agn.set_index("Subject"), per.set_index("Subject")
    both = agn.loc[~agn.single_class, "Subject"]
    da = pi.loc[both, "AUC"].values - ai.loc[both, "AUC"].values
    print(f"  delta: ROC-AUC {p_auc - a_auc:+.4f}  Bal.Acc {p_bacc - a_bacc:+.4f}  F1 {p_f1 - a_f1:+.4f}")
    print(f"  ROC-AUC improved on {int((da > 0).sum())}/{len(da)} subjects (median {np.median(da):+.4f})")
    try:
        w = stats.wilcoxon(pi.loc[both, "AUC"].values, ai.loc[both, "AUC"].values,
                           alternative="two-sided").pvalue
        print(f"  paired Wilcoxon on ROC-AUC: p = {w:.4f}")
    except Exception as e:
        print(f"  paired Wilcoxon skipped: {e}")

    tab = ai[["single_class"]].copy()
    tab["AUC_agnostic"], tab["AUC_personalized"] = ai["AUC"], pi["AUC"]
    tab["BalAcc_agnostic"], tab["BalAcc_personalized"] = ai["BalAcc"], pi["BalAcc"]
    tab.round(4).to_csv(OUT / "personalized_ebm_per_subject.csv")
    return agn, per


def _resolve_feature(token, feature_cols):
    token = token.strip()
    if token.startswith("feature_"):
        idx = int(token.split("_")[1])
        if idx < len(feature_cols):
            return feature_cols[idx]
    return token


def ebm_native_importance(X, y, feature_cols):
    """Term importances of an EBM fit on all windows (main effects and pairs)."""
    Xs = StandardScaler().fit_transform(X)
    ebm = ExplainableBoostingClassifier(random_state=SEED, n_jobs=-1).fit(Xs, y)
    imp = pd.DataFrame({"term": list(ebm.term_names_), "importance": list(ebm.term_importances())})
    imp["feature"] = imp["term"].map(
        lambda t: " & ".join(_resolve_feature(p, feature_cols) for p in t.split("&")))
    imp["label"] = imp["term"].map(
        lambda t: " × ".join(FEATURE_LABELS.get(_resolve_feature(p, feature_cols),
                                                     _resolve_feature(p, feature_cols))
                                  for p in t.split("&")))
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    imp[["feature", "label", "importance"]].to_csv(OUT / "ebm_native_importance.csv", index=False)
    print("\nEBM term importances (top 10):")
    print(imp[["feature", "importance"]].head(10).to_string(index=False))
    return imp


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    np.random.seed(SEED)
    X, y, groups, feats = load_data()
    print("\nLOSO ...")
    folds, ood = run_loso(X, y, groups)
    folds.to_csv(OUT / "folds_corrected.csv", index=False)
    summarise(folds, ood)
    run_stats(folds)
    run_personalized_ebm(X, y, groups)
    ebm_native_importance(X, y, feats)
    print("\nResults written to", OUT)
