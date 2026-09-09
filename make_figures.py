"""
Figures of the paper, drawn from the CSV files in results/ (no training).

  fig3_ebm_native_importance.pdf   EBM term importances
  fig4_generalization_gap.pdf      random split -> LOSO accuracy -> balanced accuracy -> chance
  fig5_personalized_boxplot.pdf    per-subject ROC-AUC, global scaling vs per-subject z-score

    python make_figures.py [--copy-to DIR]
"""
import argparse
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
})
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
BLUE, GREY, RED = "#3b6ea5", "#7f7f7f", "#b03a2e"


def save(fig, name, copy_to):
    fig.savefig(OUT / name, bbox_inches="tight")
    if copy_to:
        shutil.copyfile(OUT / name, Path(copy_to) / name)
    plt.close(fig)
    print("wrote", name)


def fig_importance(copy_to):
    imp = pd.read_csv(OUT / "ebm_native_importance.csv")
    top = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    ax.barh(top["label"], top["importance"], color=BLUE)
    ax.set_xlabel("Mean absolute score contribution (native EBM term importance)")
    ax.set_xlim(0, 1.05)
    save(fig, "fig3_ebm_native_importance.pdf", copy_to)


def fig_gap(copy_to):
    summ = pd.read_csv(OUT / "summary_corrected.csv")
    rs = pd.read_csv(OUT / "random_split_window_level_summary.csv", index_col=0)
    n_models = len(summ)
    cnn = OUT / "cnn_loso_summary.csv"
    cnn_rs = OUT / "cnn_random_split_summary.csv"
    if cnn.exists() and cnn_rs.exists():
        summ = pd.concat([summ, pd.read_csv(cnn)], ignore_index=True)
        rs = pd.concat([rs, pd.read_csv(cnn_rs, index_col=0)])
        n_models = len(summ)
    ebm = summ.Model == "EBM (Glassbox)"

    def stage(vals, ebm_val):
        return dict(lo=float(np.min(vals)), hi=float(np.max(vals)), ebm=float(ebm_val))

    stages = [
        stage(rs["Accuracy_mean"].values, rs.loc["EBM (Glassbox)", "Accuracy_mean"]),
        stage(summ["Acc_2class"].values, summ.loc[ebm, "Acc_2class"].item()),
        stage(summ["BalAcc_mean"].values, summ.loc[ebm, "BalAcc_mean"].item()),
        dict(lo=0.5, hi=0.5, ebm=0.5),
    ]
    labels = ["Random split\n(window-level,\n5-fold, subject-blind)\naccuracy",
              "Strict LOSO\naccuracy\n(13 subjects)",
              "Strict LOSO\nbalanced\naccuracy",
              "Chance\n(balanced\naccuracy)"]

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for i, s in enumerate(stages):
        ax.bar(i, s["hi"], width=0.58, color=BLUE if i < len(stages) - 1 else GREY, alpha=0.85, zorder=2)
        if s["hi"] - s["lo"] > 1e-6:
            ax.plot([i, i], [s["lo"], s["hi"]], color="black", lw=1.4, zorder=4)
            ax.plot([i - 0.12, i + 0.12], [s["lo"], s["lo"]], color="black", lw=1.4, zorder=4)
            ax.plot([i - 0.12, i + 0.12], [s["hi"], s["hi"]], color="black", lw=1.4, zorder=4)
            ax.text(i + 0.33, s["hi"], f"max {s['hi']:.3f}", va="center", ha="left", fontsize=7.5)
            ax.text(i + 0.33, s["lo"], f"min {s['lo']:.3f}", va="center", ha="left", fontsize=7.5)
        else:
            ax.text(i + 0.33, s["hi"], f"{s['hi']:.3f}", va="center", ha="left", fontsize=7.5)
        ax.plot(i, s["ebm"], marker="D", color=RED, ms=5, zorder=5)
    for i in range(len(stages) - 1):
        ax.annotate("", xy=(i + 0.72, stages[i + 1]["hi"]), xytext=(i + 0.30, stages[i]["hi"]),
                    arrowprops=dict(arrowstyle="->", color=GREY, lw=0.9), zorder=3)
    ax.axhline(0.5, ls="--", color=GREY, lw=0.8, zorder=1)
    ax.axhline(0.99, ls=":", color=RED, lw=1.0, zorder=1)
    ax.text(len(stages) - 0.55, 0.975, "literature-reported under randomized splits (>0.99)",
            ha="right", va="top", fontsize=7.5, color=RED)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.4, 1.03)
    ax.set_ylabel("Score")
    ax.plot([], [], marker="D", color=RED, ls="none", ms=5, label="EBM")
    words = {9: "nine", 10: "ten"}
    ax.plot([], [], color="black", lw=1.4, label=f"range over {words.get(n_models, n_models)} models")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86), ncol=2, frameon=False, columnspacing=1.2)
    save(fig, "fig4_generalization_gap.pdf", copy_to)


def fig_personalized(copy_to):
    per = pd.read_csv(OUT / "personalized_ebm_per_subject.csv")
    per = per[~per.single_class].reset_index(drop=True)
    a, p = per["AUC_agnostic"].values, per["AUC_personalized"].values
    fig, ax = plt.subplots(figsize=(4.3, 3.0))
    ax.boxplot([a, p], positions=[0, 1], widths=0.42, showfliers=False,
               medianprops=dict(color="black", lw=1.2),
               boxprops=dict(color=BLUE), whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE))
    for ai, pi, subj in zip(a, p, per["Subject"]):
        ax.plot([0, 1], [ai, pi], color=GREY, lw=0.8, alpha=0.7, zorder=2)
        ax.plot([0, 1], [ai, pi], marker="o", ms=3.5, color=BLUE, ls="none", zorder=3)
        if subj in ("s8", "s6", "g2", "s7"):
            dl = {"s8": 0.015, "g2": -0.015}.get(subj, 0.0)
            dr = {"s6": 0.022, "g2": -0.022}.get(subj, 0.0)
            ax.text(1.06, pi + dr, subj, va="center", fontsize=7.5)
            ax.text(-0.06, ai + dl, subj, va="center", ha="right", fontsize=7.5)
    ax.axhline(0.5, ls="--", color=GREY, lw=0.8, zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xlim(-0.55, 1.55)
    ax.set_xticklabels(["Patient-agnostic\n(global train-fold scaling)", "Per-subject z-score\n(unsupervised)"])
    ax.set_ylabel("Per-subject ROC-AUC (EBM, LOSO)")
    ax.set_ylim(0.05, 1.0)
    ax.set_title(f"mean {a.mean():.3f}  vs  {p.mean():.3f}   (improved on {(p > a).sum()}/13 subjects)",
                 fontsize=8.5)
    save(fig, "fig5_personalized_boxplot.pdf", copy_to)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy-to", default=None, help="also copy the PDFs into this directory")
    args = ap.parse_args()
    fig_importance(args.copy_to)
    fig_gap(args.copy_to)
    fig_personalized(args.copy_to)
