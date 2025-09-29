import os, json, time, argparse, random
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.decomposition import PCA as SKPCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, average_precision_score, roc_auc_score,
                             precision_recall_curve, roc_curve, confusion_matrix,
                             classification_report)

import matplotlib.pyplot as plt

# ------------------------- Repro ---------------------------------------------
def set_seed(seed=1234):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------- Models --------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,16,5,padding=2), nn.ReLU(), nn.MaxPool2d(2),   # 301->150
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2), # 150->75
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*75*75, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    def forward(self,x): return self.head(self.conv(x))

class TinyCNN(nn.Module):
    def __init__(self, in_ch=1, num_classes=2, p_drop=0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.AvgPool2d(2),                       # 301->150
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.AvgPool2d(2),                       # 150->75
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 64x1x1
            nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.head(self.features(x))

class TorchLogReg(nn.Module):
    """Multinomial logistic regression on flattened input."""
    def __init__(self, in_shape, n_classes=2):
        super().__init__()
        c,h,w = in_shape
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(c*h*w, n_classes)
        nn.init.zeros_(self.fc.bias)
        nn.init.normal_(self.fc.weight, 0.0, 0.01)
    def forward(self, x): return self.fc(self.flatten(x))

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# ------------------------- Plots ---------------------------------------------
def plot_history(run_dir, hist):
    es  = [h['epoch'] for h in hist]
    tl  = [h['train_loss'] for h in hist]
    vf1 = [h['val_f1'] for h in hist]
    plt.figure(); plt.plot(es, tl); plt.xlabel("Epoch"); plt.ylabel("Train loss")
    plt.title("Training loss"); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "train_loss.png")); plt.close()
    plt.figure(); plt.plot(es, vf1); plt.xlabel("Epoch"); plt.ylabel("Val Macro-F1")
    plt.title("Validation Macro-F1"); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "val_f1.png")); plt.close()

def plot_confusion_matrix(cm, class_names, out_path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(cm, interpolation='nearest'); ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label', title=title)
    thresh = cm.max()/2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)

def plot_pr_roc(y_true, y_score, run_dir):
    from sklearn.metrics import precision_recall_curve, roc_curve
    p, r, _ = precision_recall_curve(y_true, y_score)
    plt.figure(); plt.plot(r, p); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall"); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "pr_curve.png")); plt.close()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC"); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "roc_curve.png")); plt.close()

# ------------------------- Eval ----------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps, confs, probs = [], [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        lg = model(xb)
        pr = torch.softmax(lg, dim=1)              # (N,2)
        pred = pr.argmax(dim=1).cpu().numpy()
        conf = pr.max(dim=1).values.cpu().numpy()
        ys.append(yb.numpy()); ps.append(pred)
        confs.append(conf); probs.append(pr.cpu().numpy())
    y = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    probs = np.concatenate(probs)           # (N,2)
    confs = np.concatenate(confs)           # (N,)
    pos = probs[:,1]
    # robust AUCs
    try: auc_pr  = average_precision_score(y, pos)
    except: auc_pr = float('nan')
    try: auc_roc = roc_auc_score(y, pos)
    except: auc_roc = float('nan')
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "macro_f1": f1_score(y, y_pred, average="macro"),
        "mcc": matthews_corrcoef(y, y_pred),
        "auc_pr": auc_pr,
        "roc_auc": auc_roc
    }
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    return y, y_pred, probs, confs, metrics, cm, report

def evaluate_sklearn_classifier(y_true, y_pred, pos_scores):
    """Compute metrics from numpy arrays for sklearn models."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "auc_pr": average_precision_score(y_true, pos_scores) if len(np.unique(y_true)) > 1 else float('nan'),
        "roc_auc": roc_auc_score(y_true, pos_scores) if len(np.unique(y_true)) > 1 else float('nan'),
    }
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return metrics, cm, report

# ------------------------- Data helpers --------------------------------------
def load_xy_torch(base):
    Xtr = torch.load(os.path.join(base,"X_tr.pt")).float()
    ytr = torch.load(os.path.join(base,"y_tr.pt")).long()
    Xva = torch.load(os.path.join(base,"X_val.pt")).float()
    yva = torch.load(os.path.join(base,"y_val.pt")).long()
    Xte = torch.load(os.path.join(base,"X_test.pt")).float()
    yte = torch.load(os.path.join(base,"y_test.pt")).long()
    return Xtr, ytr, Xva, yva, Xte, yte

def apply_channelwise_standardize(X_train, X_val, X_test):
    with torch.no_grad():
        mu = X_train.mean(dim=(0,2,3), keepdim=True)
        sd = X_train.std(dim=(0,2,3), keepdim=True).clamp_min(1e-6)
        return (X_train-mu)/sd, (X_val-mu)/sd, (X_test-mu)/sd

def apply_pca_for_logreg(Xtr, Xva, Xte, pca_arg, seed, standardize):
    """
    Flatten -> (optional) StandardScaler -> PCA -> back to torch as (N,1,1,K)
    Only for logreg (CNNs expect images).
    pca_arg: float in (0,1] for variance (e.g., 0.95) or int for components.
    """
    # flatten to 2D
    Ntr, C, H, W = Xtr.shape
    D = C*H*W
    Xtr_np = Xtr.reshape(Ntr, D).cpu().numpy()
    Xva_np = Xva.reshape(Xva.shape[0], D).cpu().numpy()
    Xte_np = Xte.reshape(Xte.shape[0], D).cpu().numpy()

    scaler = None
    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xtr_np = scaler.fit_transform(Xtr_np)
        Xva_np = scaler.transform(Xva_np)
        Xte_np = scaler.transform(Xte_np)

    if isinstance(pca_arg, float):
        if not (0 < pca_arg <= 1.0): raise ValueError("--pca must be in (0,1] if float")
        pca = SKPCA(n_components=pca_arg, svd_solver="full", random_state=seed)
    else:  # int
        if pca_arg < 1: raise ValueError("--pca int must be >=1")
        pca = SKPCA(n_components=pca_arg, svd_solver="randomized", random_state=seed)

    Xtr_red = pca.fit_transform(Xtr_np).astype(np.float32)
    Xva_red = pca.transform(Xva_np).astype(np.float32)
    Xte_red = pca.transform(Xte_np).astype(np.float32)

    # back to torch with shape (N,1,1,K) so TorchLogReg (flatten) just works
    def to_tensor(X):
        t = torch.from_numpy(X)
        return t.view(t.shape[0], 1, 1, t.shape[1]).contiguous()

    return to_tensor(Xtr_red), to_tensor(Xva_red), to_tensor(Xte_red), pca

def prep_features_for_svm(Xtr, Xva, Xte, pca_arg, seed, standardize):
    """
    Returns 2D numpy arrays suitable for sklearn SVC:
      - If pca_arg==0: Flatten (+ optional StandardScaler).
      - Else: Flatten -> (optional) StandardScaler -> PCA (variance or components).
    Also returns a dict with 'scaler' and 'pca' (any may be None).
    """
    Ntr, C, H, W = Xtr.shape
    D = C*H*W
    Xtr_np = Xtr.reshape(Ntr, D).cpu().numpy()
    Xva_np = Xva.reshape(Xva.shape[0], D).cpu().numpy()
    Xte_np = Xte.reshape(Xte.shape[0], D).cpu().numpy()

    scaler = None
    pca = None

    if isinstance(pca_arg, str):
        # accept "0" as off
        pca_arg = 0 if pca_arg.strip() == "0" else pca_arg

    if pca_arg == 0:
        if standardize:
            scaler = StandardScaler(with_mean=True, with_std=True)
            Xtr_np = scaler.fit_transform(Xtr_np)
            Xva_np = scaler.transform(Xva_np)
            Xte_np = scaler.transform(Xte_np)
    else:
        if isinstance(pca_arg, int):
            if pca_arg < 1: raise ValueError("--pca int must be >=1")
            scaler = StandardScaler(with_mean=True, with_std=True) if standardize else None
            if scaler is not None:
                Xtr_np = scaler.fit_transform(Xtr_np)
                Xva_np = scaler.transform(Xva_np)
                Xte_np = scaler.transform(Xte_np)
            pca = SKPCA(n_components=pca_arg, svd_solver="randomized", random_state=seed)
        else:
            pca_arg = float(pca_arg)
            if not (0 < pca_arg <= 1.0): raise ValueError("--pca must be in (0,1] if float")
            scaler = StandardScaler(with_mean=True, with_std=True) if standardize else None
            if scaler is not None:
                Xtr_np = scaler.fit_transform(Xtr_np)
                Xva_np = scaler.transform(Xva_np)
                Xte_np = scaler.transform(Xte_np)
            pca = SKPCA(n_components=pca_arg, svd_solver="full", random_state=seed)

        Xtr_np = pca.fit_transform(Xtr_np).astype(np.float32)
        Xva_np = pca.transform(Xva_np).astype(np.float32)
        Xte_np = pca.transform(Xte_np).astype(np.float32)

    return Xtr_np.astype(np.float32), Xva_np.astype(np.float32), Xte_np.astype(np.float32), {"scaler": scaler, "pca": pca}

# ------------------------- Main ----------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="logreg",
                    choices=["logreg","cnn","tinycnn","svm"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--standardize", action="store_true",
                    help="Channel-wise z-score for image inputs; for PCA+logreg/SVM, feature-wise z-score.")
    ap.add_argument("--pca", type=str, default="0",
                    help="0=off; float in (0,1] is variance target (e.g., 0.95); int >=1 is #components.")
    # SVM args
    ap.add_argument("--svm-kernel", type=str, default="rbf",
                    choices=["linear","rbf","poly"],
                    help="Kernel for SVM when --model svm.")
    ap.add_argument("--svm-C", type=float, default=1.0,
                    help="C regularization for SVM.")
    ap.add_argument("--svm-gamma", type=str, default="scale",
                    help="Gamma for RBF/poly kernels: {'scale','auto'} or a float (as str).")
    ap.add_argument("--svm-degree", type=int, default=3,
                    help="Degree for poly kernel.")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = os.getcwd()

    # Load data
    Xtr, ytr, Xva, yva, Xte, yte = load_xy_torch(base)

    # --- SVM path (sklearn) ---------------------------------------------------
    if args.model == "svm":
        # Prepare 2D features for sklearn SVC (handles flatten, optional standardize, optional PCA)
        Xtr_np, Xva_np, Xte_np, xf = prep_features_for_svm(
            Xtr, Xva, Xte, args.pca, args.seed, standardize=args.standardize
        )
        ytr_np = ytr.cpu().numpy()
        yva_np = yva.cpu().numpy()
        yte_np = yte.cpu().numpy()

        # Parse gamma if numeric provided as string
        gamma = args.svm_gamma
        try:
            gamma = float(gamma)
        except ValueError:
            # keep 'scale' or 'auto'
            pass

        # Class imbalance handling analogous to CrossEntropyLoss weights
        # Use 'balanced' so class weights are inversely proportional to class frequencies.
        svc = SVC(
            kernel=args.svm_kernel,
            C=args.svm_C,
            gamma=gamma if args.svm_kernel in ("rbf","poly") else "scale",
            degree=args.svm_degree,
            probability=True,          # enables predict_proba for PR/ROC
            class_weight="balanced",
            random_state=args.seed,
        )

        # Fit once using provided hyperparameters.
        t0 = time.time()
        svc.fit(Xtr_np, ytr_np)
        fit_time = time.time() - t0

        # Validation report (for parity with torch loop logs)
        va_probs = svc.predict_proba(Xva_np)
        va_pred  = va_probs.argmax(1)
        va_metrics, _, _ = evaluate_sklearn_classifier(yva_np, va_pred, va_probs[:,1])
        print(f"SVM | kernel={args.svm_kernel} C={args.svm_C} gamma={args.svm_gamma} degree={args.svm_degree}")
        print(f"Validation: acc={va_metrics['accuracy']:.3f} macroF1={va_metrics['macro_f1']:.3f}")

        # Create run dir
        run_id = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base, "runs", f"svm_{run_id}")
        os.makedirs(run_dir, exist_ok=True)

        # Save a tiny 'history' with one entry to keep artifacts consistent
        history = [{"epoch": 1, "train_loss": float('nan'), "val_acc": va_metrics["accuracy"], "val_f1": va_metrics["macro_f1"]}]
        with open(os.path.join(run_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        # still produce the same plots (they'll be trivial)
        plot_history(run_dir, history)

        # Test
        te_probs = svc.predict_proba(Xte_np)
        te_pred  = te_probs.argmax(1)
        metrics, cm, report = evaluate_sklearn_classifier(yte_np, te_pred, te_probs[:,1])

        # Emulate "total/trainable params" notion (SVM has support vectors)
        total_p = train_p = int(svc.support_vectors_.size)

        # Print metrics (same style as torch branch)
        print("\n=== Test Metrics ===")
        for k, v in metrics.items():
            print(f"{k.replace('_',' ').title():18s}: {v:.4f}")
        print("\n=== Per-class PRF ===")
        for cls in range(2):
            prf = report[str(cls)]
            print(f"class {cls}: precision={prf['precision']:.4f} recall={prf['recall']:.4f} f1={prf['f1-score']:.4f}")
        print(f"\nParams: total={total_p:,}  trainable={train_p:,}")

        # Save artifacts
        with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
            json.dump({"metrics": metrics, "report": report, "params": {"total": int(total_p), "trainable": int(train_p)}}, f, indent=2)

        class_names = [f"Class {i}" for i in range(2)]
        plot_confusion_matrix(cm, class_names, os.path.join(run_dir, "confusion_matrix.png"))
        plot_pr_roc(yte_np, te_probs[:,1], run_dir)

        # Per-example CSV
        out_csv = os.path.join(run_dir, "predictions.csv")
        with open(out_csv, "w") as f:
            f.write("true,pred,prob_0,prob_1,confidence\n")
            confs = te_probs.max(axis=1)
            for i in range(len(yte_np)):
                f.write(f"{int(yte_np[i])},{int(te_pred[i])},{te_probs[i,0]:.6f},{te_probs[i,1]:.6f},{float(confs[i]):.6f}\n")

        # Save PCA info if used
        pca_used = xf["pca"] is not None
        if pca_used:
            pca_obj = xf["pca"]
            with open(os.path.join(run_dir, "pca_info.json"), "w") as f:
                info = {
                    "n_components_": int(getattr(pca_obj, "n_components_", 0)) if hasattr(pca_obj, "n_components_") else None,
                    "explained_variance_ratio_sum": float(np.sum(getattr(pca_obj, "explained_variance_ratio_", [0]))),
                }
                json.dump(info, f, indent=2)

        # Short summary
        with open(os.path.join(run_dir, "summary.txt"), "w") as f:
            f.write("Test Summary\n")
            f.write(f"Run dir: {run_dir}\n")
            f.write(f"Model: svm (kernel={args.svm_kernel}, C={args.svm_C}, gamma={args.svm_gamma}, degree={args.svm_degree})\n")
            for k, v in metrics.items(): f.write(f"{k}: {v:.4f}\n")
            f.write(f"params_total: {total_p}\nparams_trainable: {train_p}\n")
            if pca_used: f.write("PCA applied (see pca_info.json)\n")
            f.write(f"fit_time_sec: {fit_time:.3f}\n")

        print("\nDone. Artifacts saved to:", run_dir)
        return  # IMPORTANT: skip the PyTorch path

    # ------------------------- (original PyTorch path continues) ---------------
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base, "runs", f"{args.model}_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    # Standardize / PCA handling
    pca_used = False
    if args.model == "logreg" and args.pca.strip() != "0":
        # parse pca arg
        try: pca_arg = int(args.pca)
        except ValueError: pca_arg = float(args.pca)
        Xtr, Xva, Xte, pca_obj = apply_pca_for_logreg(Xtr, Xva, Xte, pca_arg, args.seed, standardize=args.standardize)
        pca_used = True
    else:
        if args.standardize:
            Xtr, Xva, Xte = apply_channelwise_standardize(Xtr, Xva, Xte)

    # Loaders
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.bs, shuffle=True,  pin_memory=True, num_workers=0)
    val_loader   = DataLoader(TensorDataset(Xva, yva), batch_size=2*args.bs, shuffle=False, pin_memory=True, num_workers=0)
    test_loader  = DataLoader(TensorDataset(Xte, yte), batch_size=2*args.bs, shuffle=False, pin_memory=True, num_workers=0)

    # Model
    in_shape = tuple(Xtr.shape[1:])  # (C, H, W) or (1,1,K) after PCA
    if args.model == "cnn":
        if pca_used: raise ValueError("PCA is only supported for --model logreg.")
        model = SimpleCNN().to(device)
    elif args.model == "tinycnn":
        if pca_used: raise ValueError("PCA is only supported for --model logreg.")
        model = TinyCNN().to(device)
    else:
        model = TorchLogReg(in_shape=in_shape, n_classes=2).to(device)

    total_p, train_p = count_params(model)

    # Class weights
    counts = np.bincount(ytr.cpu().numpy(), minlength=2).astype(np.float32)
    inv = 1.0 / np.maximum(counts, 1)
    w = (inv / inv.sum()) * 2.0
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Train loop with early stopping on val macro-F1
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best_metric, wait = -1.0, 0
    best_path = os.path.join(run_dir, "best.pt")
    history = []

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            total_loss += float(loss.item())

        # Validation
        _, _, _, _, val_metrics, _, _ = evaluate(model, val_loader, device)
        val_acc, val_f1 = val_metrics["accuracy"], val_metrics["macro_f1"]
        history.append({"epoch": epoch, "train_loss": total_loss, "val_acc": val_acc, "val_f1": val_f1})
        print(f"Epoch {epoch:03d} | train_loss={total_loss:.3f} | val_acc={val_acc:.3f} | val_macroF1={val_f1:.3f}")
        scheduler.step(val_f1)

        # Early stopping
        if val_f1 > best_metric + 1e-6:
            best_metric, wait = val_f1, 0
            torch.save(model.state_dict(), best_path)
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stopping.")
                break

    # Save history plots
    with open(os.path.join(run_dir, "history.json"), "w") as f: json.dump(history, f, indent=2)
    plot_history(run_dir, history)

    # Test
    model.load_state_dict(torch.load(best_path, map_location=device))
    y, y_pred, probs, confs, metrics, cm, report = evaluate(model, test_loader, device)
    pos_prob = probs[:,1]
    class_names = [f"Class {i}" for i in range(2)]

    # Print metrics
    print("\n=== Test Metrics ===")
    for k,v in metrics.items(): print(f"{k.replace('_',' ').title():18s}: {v:.4f}")
    print("\n=== Per-class PRF ===")
    for cls in range(2):
        prf = report[str(cls)]
        print(f"class {cls}: precision={prf['precision']:.4f} recall={prf['recall']:.4f} f1={prf['f1-score']:.4f}")
    print(f"\nParams: total={total_p:,}  trainable={train_p:,}")

    # Save artifacts
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump({"metrics":metrics,"report":report,"params":{"total":int(total_p),"trainable":int(train_p)}}, f, indent=2)

    plot_confusion_matrix(cm, class_names, os.path.join(run_dir, "confusion_matrix.png"))
    plot_pr_roc(y, pos_prob, run_dir)

    # Per-example CSV
    out_csv = os.path.join(run_dir, "predictions.csv")
    with open(out_csv, "w") as f:
        f.write("true,pred,prob_0,prob_1,confidence\n")
        for i in range(len(y)):
            f.write(f"{int(y[i])},{int(y_pred[i])},{probs[i,0]:.6f},{probs[i,1]:.6f},{float(confs[i]):{'.6f'}}\n")

    # Save PCA info if used
    if pca_used:
        with open(os.path.join(run_dir, "pca_info.json"), "w") as f:
            info = {
                "n_components_": int(pca_obj.n_components_) if hasattr(pca_obj, "n_components_") else None,
                "explained_variance_ratio_sum": float(np.sum(getattr(pca_obj, "explained_variance_ratio_", [0]))),
            }
            json.dump(info, f, indent=2)

    # Short summary
    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write("Test Summary\n")
        f.write(f"Run dir: {run_dir}\n")
        f.write(f"Model: {args.model}\n")
        for k,v in metrics.items(): f.write(f"{k}: {v:.4f}\n")
        f.write(f"params_total: {total_p}\nparams_trainable: {train_p}\n")
        if pca_used: f.write("PCA applied (see pca_info.json)\n")

    print("\nDone. Artifacts saved to:", run_dir)

if __name__ == "__main__":
    main()
