import argparse, json
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, average_precision_score, roc_auc_score,
                             classification_report, confusion_matrix)

def load_predictions(paths):
    """Load list of CSVs, each with columns true,pred,prob_0,prob_1,confidence"""
    dfs = [pd.read_csv(p) for p in paths]
    y_true = dfs[0]['true'].to_numpy()
    probs = [df['prob_1'].to_numpy() for df in dfs]
    return y_true, probs

def evaluate_ensemble(y, probs_list, weights=None):
    probs_arr = np.vstack(probs_list)  # shape (M, N)
    M, N = probs_arr.shape
    if weights is None:
        weights = np.ones(M) / M
    weights = np.array(weights) / np.sum(weights)
    pos_prob = np.average(probs_arr, axis=0, weights=weights)
    y_pred = (pos_prob >= 0.5).astype(int)

    # metrics
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "macro_f1": f1_score(y, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y, y_pred),
        "auc_pr": average_precision_score(y, pos_prob),
        "roc_auc": roc_auc_score(y, pos_prob)
    }
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    return y_pred, pos_prob, metrics, cm, report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="Paths to predictions.csv files")
    ap.add_argument("--weights", type=float, nargs="*", help="Optional weights (same order as csvs)")
    args = ap.parse_args()

    y, probs_list = load_predictions(args.csvs)
    y_pred, pos_prob, metrics, cm, report = evaluate_ensemble(y, probs_list, args.weights)

    print("\n=== Ensemble Metrics ===")
    for k,v in metrics.items():
        print(f"{k.replace('_',' ').title():18s}: {v:.4f}")

    print("\n=== Per-class PRF ===")
    for cls in range(2):
        prf = report[str(cls)]
        print(f"class {cls}: precision={prf['precision']:.4f} "
              f"recall={prf['recall']:.4f} f1={prf['f1-score']:.4f}")

    print("\nConfusion Matrix:\n", cm)

    # Save to JSON for later
    out = {"metrics":metrics, "report":report, "confusion_matrix":cm.tolist()}
    with open("ensemble_metrics.json","w") as f: json.dump(out, f, indent=2)
    out_df = pd.DataFrame({
        "true": y,
        "pred": y_pred,
        "prob_0": 1 - pos_prob,
        "prob_1": pos_prob,
        "confidence": np.maximum(pos_prob, 1 - pos_prob)
    })
    out_df.to_csv("predictions_ensemble.csv", index=False)
    print("Saved ensemble predictions to predictions_ensemble.csv")
if __name__ == "__main__":
    main()

