#!/usr/bin/env python3
"""
make_diagnostics.py

Usage:
  python make_diagnostics.py --input predictions.csv --outdir output_dir

Reads predictions.csv with columns:
  true,pred,prob_0,prob_1,confidence
Appends simulation + inclination, writes predictions_with_incl.csv,
and generates a suite of diagnostic plots (no titles).
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report, brier_score_loss
)
from sklearn.calibration import calibration_curve

# ---------- Edit this list if your order/counts change ----------
# (sim_id, n_rows, inclination_deg)
SIM_BLOCKS = [
    ("sim10", 11, 8.10),
    ("sim04", 11, 15.0),
    ("sim37",  3, 1.73),
    ("sim47", 11, 14.3),
    ("sim39", 11, 20.3),
]
# ----------------------------------------------------------------

def attach_inclinations(df: pd.DataFrame) -> pd.DataFrame:
    sim_ids, incl = [], []
    for sim, n, inc in SIM_BLOCKS:
        sim_ids.extend([sim] * n)
        incl.extend([inc] * n)
    if len(sim_ids) != len(df):
        raise ValueError(
            f"Row count mismatch: predictions has {len(df)} rows, "
            f"but SIM_BLOCKS sum to {len(sim_ids)}. "
            "Edit SIM_BLOCKS to match file order + counts."
        )
    out = df.copy()
    out["simulation"] = sim_ids
    out["inclination"] = incl
    return out

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

# --------- Plot helpers (NO TITLES) ---------
def plot_confusion_matrix(y_true, y_pred, outpath):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Class 0", "Class 1"])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

def plot_roc(y_true, y_score, outpath):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

def plot_pr(y_true, y_score, outpath):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

def plot_confidence_hist(df, outpath_all, outpath_split):
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.hist(df["confidence"], bins=20, edgecolor="black")
    ax.set_xlabel("Prediction Confidence")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath_all, dpi=200)
    plt.close(fig)

    df2 = df.copy()
    df2["correct"] = (df2["true"] == df2["pred"])
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.hist(df2.loc[df2["correct"], "confidence"], bins=20, alpha=0.7, label="Correct")
    ax.hist(df2.loc[~df2["correct"], "confidence"], bins=20, alpha=0.7, label="Incorrect")
    ax.set_xlabel("Prediction Confidence")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath_split, dpi=200)
    plt.close(fig)

def plot_confidence_vs_incl(df, outpath):
    df2 = df.copy()
    df2["correct"] = (df2["true"] == df2["pred"])
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    sc = ax.scatter(df2["inclination"], df2["confidence"],
                    c=df2["correct"].map({True: 1, False: 0}),
                    cmap="coolwarm", alpha=0.75)
    means = df2.groupby("inclination")["confidence"].mean().reset_index()
    ax.plot(means["inclination"], means["confidence"], marker="o", linewidth=1.6, label="Mean confidence")
    ax.set_xlabel("Inclination (deg)")
    ax.set_ylabel("Prediction Confidence")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

def plot_prob1_by_trueclass(df, outpath):
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.hist(df.loc[df["true"] == 0, "prob_1"], bins=20, alpha=0.7, label="True class 0")
    ax.hist(df.loc[df["true"] == 1, "prob_1"], bins=20, alpha=0.7, label="True class 1")
    ax.set_xlabel("Predicted Probability of Class 1 (prob_1)")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

def plot_calibration(df, outpath):
    y_true = df["true"].astype(int).values
    y_prob = df["prob_1"].astype(float).values
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    brier = brier_score_loss(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(mean_pred, frac_pos, marker="o", label=f"Brier = {brier:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

def plot_avg_prob_vs_inclination(df, outpath):
    """
    Average predicted probability vs inclination, by true class:
      - line for Class 0: mean(1 - prob_1)
      - line for Class 1: mean(prob_1)
    """
    grouped = df.groupby("inclination")[["prob_0", "prob_1"]].mean().reset_index()

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(grouped["inclination"], grouped["prob_0"], marker="o", label="Class 0")
    ax.plot(grouped["inclination"], grouped["prob_1"], marker="o", label="Class 1")

    ax.set_xlabel("Inclination (deg)")
    ax.set_ylabel("Avg predicted probability")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)
# -------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="predictions.csv", help="Path to predictions.csv")
    parser.add_argument("--outdir", type=str, default="diagnostic_plots", help="Directory for plots + enriched CSV")
    args = parser.parse_args()

    ensure_outdir(args.outdir)

    # Load predictions
    df = pd.read_csv(args.input)
    needed = {"true", "pred", "prob_0", "prob_1", "confidence"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {args.input}: {missing}")

    # Attach simulation + inclination
    df = attach_inclinations(df)
    enriched_csv = os.path.join(args.outdir, "predictions_with_incl.csv")
    df.to_csv(enriched_csv, index=False)

    # Print a quick text report
    y_true = df["true"].astype(int).values
    y_pred = df["pred"].astype(int).values
    y_score = df["prob_1"].astype(float).values

    print("=== Classification report ===")
    print(classification_report(y_true, y_pred, digits=4))
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    ap = average_precision_score(y_true, y_score)
    print(f"ROC AUC: {roc_auc:.4f} | PR AUC: {ap:.4f}")

    # Make plots (no titles)
    plot_confusion_matrix(y_true, y_pred, os.path.join(args.outdir, "confusion_matrix.png"))
    plot_roc(y_true, y_score, os.path.join(args.outdir, "roc_curve.png"))
    plot_pr(y_true, y_score, os.path.join(args.outdir, "pr_curve.png"))
    plot_confidence_hist(df, os.path.join(args.outdir, "confidence_hist.png"),
                         os.path.join(args.outdir, "confidence_hist_split.png"))
    plot_confidence_vs_incl(df, os.path.join(args.outdir, "confidence_vs_inclination.png"))
    plot_prob1_by_trueclass(df, os.path.join(args.outdir, "prob1_by_trueclass.png"))
    plot_calibration(df, os.path.join(args.outdir, "calibration_curve.png"))
    plot_avg_prob_vs_inclination(df, os.path.join(args.outdir, "avg_probs_vs_inclination.png"))

    print(f"Saved enriched CSV to: {enriched_csv}")
    print(f"Saved plots to: {args.outdir}")

if __name__ == "__main__":
    main()

