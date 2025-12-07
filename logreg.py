import os, json
import numpy as np
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, average_precision_score, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight

def to_np32(t):
    return t.detach().cpu().numpy().astype(np.float32, copy=False)
def choose_pca_k95(X_train_flat, probe_max=512, var_target=0.95, random_state=1234):
    """
    Probe with randomized PCA up to `probe_max` comps to estimate how many comps
    reach `var_target` cumulative variance, then return that integer.
    """
    n_samples, n_features = X_train_flat.shape
    max_possible = min(n_samples - 1, n_features)
    probe_k = min(probe_max, max_possible)

    probe = PCA(
        n_components=probe_k,
        svd_solver="randomized",
        whiten=False,
        random_state=random_state
    ).fit(X_train_flat)

    cum = np.cumsum(probe.explained_variance_ratio_)
    k95 = int(np.searchsorted(cum, var_target) + 1)
    # Ensure at least 1 and at most what we probed
    k95 = max(1, min(k95, probe_k))
    return k95

def main():
    base = os.getcwd()

    # Load tensors and cast to float32
    X_tr  = to_np32(torch.load(os.path.join(base, "X_tr.pt")).float())
    y_tr  = torch.load(os.path.join(base, "y_tr.pt")).long().cpu().numpy()
    X_val = to_np32(torch.load(os.path.join(base, "X_val.pt")).float())
    y_val = torch.load(os.path.join(base, "y_val.pt")).long().cpu().numpy()
    X_te  = to_np32(torch.load(os.path.join(base, "X_test.pt")).float())
    y_te  = torch.load(os.path.join(base, "y_test.pt")).long().cpu().numpy()

    # Flatten images: (N, C, H, W) -> (N, C*H*W)
    X_tr  = X_tr.reshape(X_tr.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_te  = X_te.reshape(X_te.shape[0], -1)


    # Standardize BEFORE probing PCA
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_tr_std = scaler.fit_transform(X_tr)   # float32 → internally float64 but just once
    X_val_std = scaler.transform(X_val)
    X_te_std  = scaler.transform(X_te)
    # Pick k to hit ~95% variance without full SVD
    k95 = choose_pca_k95(X_tr_std, probe_max=512, var_target=0.99, random_state=1234)
    print(f"[PCA] Chosen components for ~95% variance: k={k95}")

    # Class weights (balanced) to handle imbalance
    classes = np.unique(y_tr)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

    pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),  # refit inside pipeline
    # ("pca", PCA(n_components=k95, svd_solver="randomized", whiten=False, random_state=1234)),
    ("clf", LogisticRegression(
        solver="lbfgs", penalty="l2", C=1.0, max_iter=2000,
        class_weight="balanced", n_jobs=1, random_state=1234
        )),
    ])


    # Fit on TRAIN only, then evaluate on VAL and TEST
    pipe.fit(X_tr, y_tr)

    def eval_split(X, y):
        p = pipe.predict(X)
        s = pipe.predict_proba(X)[:, 1]
        return {
            "accuracy": accuracy_score(y, p),
            "balanced_accuracy": balanced_accuracy_score(y, p),
            "macro_f1": f1_score(y, p, average="macro"),
            "mcc": matthews_corrcoef(y, p),
            "auc_pr": average_precision_score(y, s),
            "roc_auc": roc_auc_score(y, s),
        }

    val_metrics = eval_split(X_val, y_val)
    test_metrics = eval_split(X_te, y_te)

    print("=== VAL ===")
    for k,v in val_metrics.items(): print(f"{k}: {v:.4f}")
    print("\n=== TEST ===")
    for k,v in test_metrics.items(): print(f"{k}: {v:.4f}")

    with open("minimal_results.json", "w") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, indent=2)

if __name__ == "__main__":
    main()

