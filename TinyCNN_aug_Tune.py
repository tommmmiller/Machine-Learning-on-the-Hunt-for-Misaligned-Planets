import os, json, time, argparse, random
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, average_precision_score, roc_auc_score,
                             precision_recall_curve, roc_curve, confusion_matrix,
                             classification_report)
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

# ------------------------- Repro ---------------------------------------------
def set_seed(seed=1234):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------- Model (TinyCNN) -----------------------------------
class TinyCNN(nn.Module):
    """
    3 Conv blocks (BN+ReLU+MaxPool on first two), then Conv, GAP head.
    Width via base_w; compact but expressive.
    """
    def __init__(self, in_ch=1, num_classes=2, base_w=32, p_drop=0.3):
        super().__init__()
        c1, c2, c3 = base_w, base_w*2, base_w*4
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, c1, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 301->150
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 150->75
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(c3, num_classes)
        )
    def forward(self, x): return self.head(self.features(x))

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

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

# ------------------------- Augmentation --------------------------------------
class RandomRotate:
    def __init__(self, degrees=15.0, fill=0.0, interpolation="bilinear"):
        self.degrees = float(degrees)
        self.fill = float(fill)
        self.interp = InterpolationMode.BILINEAR if interpolation=="bilinear" else InterpolationMode.NEAREST
    def __call__(self, x):
        if self.degrees <= 0: return x
        angle = random.uniform(-self.degrees, self.degrees)
        return TF.rotate(x, angle=angle, interpolation=self.interp, expand=False, fill=self.fill)

class RandomFlips:
    def __init__(self, p_h=0.5, p_v=0.5):
        self.p_h, self.p_v = float(p_h), float(p_v)
    def __call__(self, x):
        if random.random() < self.p_h:
            x = TF.hflip(x)
        if random.random() < self.p_v:
            x = TF.vflip(x)
        return x

class ComposeT:
    def __init__(self, transforms): self.transforms = transforms
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class AugmentedTensorDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X, self.y, self.t = X, y, transform
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        xi = self.X[i]
        if self.t is not None:
            xi = self.t(xi)
        return xi, self.y[i]

# ------------------------- Eval / Plots --------------------------------------
@torch.no_grad()
def forward_probs(model, loader, device):
    model.eval()
    ys, probs = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        pr = torch.softmax(model(xb), dim=1).cpu().numpy()
        ys.append(yb.numpy()); probs.append(pr)
    y = np.concatenate(ys)
    probs = np.concatenate(probs)  # (N,2)
    return y, probs

def metrics_from_preds(y_true, y_pred, pos_prob):
    try: auc_pr  = average_precision_score(y_true, pos_prob)
    except: auc_pr = float('nan')
    try: auc_roc = roc_auc_score(y_true, pos_prob)
    except: auc_roc = float('nan')
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "auc_pr": auc_pr,
        "roc_auc": auc_roc
    }
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return metrics, cm, report

def evaluate_argmax(model, loader, device):
    y, probs = forward_probs(model, loader, device)
    y_pred = probs.argmax(axis=1)
    metrics, cm, report = metrics_from_preds(y, y_pred, probs[:,1])
    return (metrics, cm, report), y, y_pred, probs

def metric_from_threshold(y_true, pos_prob, metric="macro_f1", thr=0.5):
    yhat = (pos_prob >= thr).astype(int)
    if metric == "macro_f1":
        return f1_score(y_true, yhat, average="macro", zero_division=0)
    elif metric == "balanced_accuracy":
        return balanced_accuracy_score(y_true, yhat)
    elif metric == "f1_pos":
        return f1_score(y_true, yhat, pos_label=1, zero_division=0)
    else:
        raise ValueError("Unknown tune metric")

def tune_threshold(y_true, pos_prob, metric="macro_f1", steps=1001):
    ts = np.linspace(0.0, 1.0, steps)
    scores = [metric_from_threshold(y_true, pos_prob, metric, t) for t in ts]
    best_idx = int(np.argmax(scores))
    return float(ts[best_idx]), float(scores[best_idx])

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

def plot_pr_roc(y_true, y_score, run_dir, suffix=""):
    p, r, _ = precision_recall_curve(y_true, y_score)
    plt.figure(); plt.plot(r, p); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision-Recall{suffix}"); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"pr_curve{suffix}.png")); plt.close()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC{suffix}"); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"roc_curve{suffix}.png")); plt.close()

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

# ------------------------- Main ----------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--standardize", action="store_true",
                    help="Channel-wise z-score for image inputs.")
    # TinyCNN args
    ap.add_argument("--tiny-width", type=int, default=32, help="Base width (channels) for TinyCNN (try 48).")
    ap.add_argument("--tiny-dropout", type=float, default=0.3, help="Dropout in TinyCNN head.")
    # Label smoothing
    ap.add_argument("--label-smoothing", type=float, default=0.0, help="CE label smoothing (e.g., 0.05).")
    # Augmentation args
    ap.add_argument("--rot-degrees", type=float, default=15.0, help="Max absolute rotation in degrees.")
    ap.add_argument("--flip-p-h", type=float, default=0.5, help="Probability of horizontal flip.")
    ap.add_argument("--flip-p-v", type=float, default=0.5, help="Probability of vertical flip.")
    ap.add_argument("--interp", type=str, default="bilinear", choices=["bilinear","nearest"], help="Rotation interpolation.")
    # Threshold tuning
    ap.add_argument("--tune-threshold", action="store_true", help="Enable validation-based threshold tuning.")
    ap.add_argument("--tune-metric", type=str, default="macro_f1",
                    choices=["macro_f1","balanced_accuracy","f1_pos"],
                    help="Metric to maximize on validation when tuning threshold.")
    ap.add_argument("--tune-steps", type=int, default=1001, help="Grid size in [0,1] for threshold search.")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = os.getcwd()

    # Run dir
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base, "runs", f"tinycnn_aug_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    # Load data
    Xtr, ytr, Xva, yva, Xte, yte = load_xy_torch(base)
    if args.standardize:
        Xtr, Xva, Xte = apply_channelwise_standardize(Xtr, Xva, Xte)

    # Augmentation (training only)
    fill_val = 0.0 if args.standardize else float(Xtr.mean().item())
    train_tf = ComposeT([
        RandomRotate(degrees=args.rot_degrees, fill=fill_val, interpolation=args.interp),
        RandomFlips(p_h=args.flip_p_h, p_v=args.flip_p_v),
    ])
    train_ds = AugmentedTensorDataset(Xtr, ytr, transform=train_tf)
    val_ds   = TensorDataset(Xva, yva)   # no augmentation
    test_ds  = TensorDataset(Xte, yte)   # no augmentation

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,  pin_memory=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=2*args.bs, shuffle=False, pin_memory=True, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=2*args.bs, shuffle=False, pin_memory=True, num_workers=0)

    # Model
    in_shape = tuple(Xtr.shape[1:])
    model = TinyCNN(in_ch=in_shape[0], base_w=args.tiny_width, p_drop=args.tiny_dropout).to(device)
    total_p, train_p = count_params(model)

    # Class weights
    counts = np.bincount(ytr.cpu().numpy(), minlength=2).astype(np.float32)
    inv = 1.0 / np.maximum(counts, 1)
    w = (inv / inv.sum()) * 2.0
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, device=device),
                                    label_smoothing=float(args.label_smoothing))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # Train loop with early stopping on val macro-F1 (argmax during training)
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

        # Validation (argmax)
        (val_metrics, _, _), _, _, probs_val = evaluate_argmax(model, val_loader, device)
        val_acc, val_f1 = val_metrics["accuracy"], val_metrics["macro_f1"]
        history.append({"epoch": epoch, "train_loss": total_loss, "val_acc": val_acc, "val_f1": val_f1})
        print(f"Epoch {epoch:03d} | train_loss={total_loss:.3f} | val_acc={val_acc:.3f} | val_macroF1={val_f1:.3f}")
        scheduler.step(val_f1)

        # Early stopping on val macro-F1
        if val_f1 > best_metric + 1e-6:
            best_metric, wait = val_f1, 0
            torch.save(model.state_dict(), best_path)
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stopping.")
                break

    # Save training history & plots
    with open(os.path.join(run_dir, "history.json"), "w") as f: json.dump(history, f, indent=2)
    plot_history(run_dir, history)

    # Load best and evaluate (argmax)
    model.load_state_dict(torch.load(best_path, map_location=device))
    (metrics_argmax, cm_argmax, report_argmax), y_te, y_pred_te, probs_te = evaluate_argmax(model, test_loader, device)
    pos_prob_te = probs_te[:,1]
    class_names = [f"Class {i}" for i in range(2)]

    print("\n=== Test Metrics (Argmax 0.5) ===")
    for k,v in metrics_argmax.items(): print(f"{k.replace('_',' ').title():18s}: {v:.4f}")
    print("\n=== Per-class PRF (Argmax) ===")
    for cls in range(2):
        prf = report_argmax[str(cls)]
        print(f"class {cls}: precision={prf['precision']:.4f} recall={prf['recall']:.4f} f1={prf['f1-score']:.4f}")
    print(f"\nParams: total={total_p:,}  trainable={train_p:,}")

    with open(os.path.join(run_dir, "test_metrics_argmax.json"), "w") as f:
        json.dump({"metrics":metrics_argmax,"report":report_argmax,"params":{"total":int(total_p),"trainable":int(train_p)}}, f, indent=2)
    plot_confusion_matrix(cm_argmax, class_names, os.path.join(run_dir, "confusion_matrix_argmax.png"))
    plot_pr_roc(y_te, pos_prob_te, run_dir, suffix="_argmax")

    # Threshold tuning (optional)
    tuned = {}
    if args.tune_threshold:
        y_va, probs_va = forward_probs(model, val_loader, device)
        pos_prob_va = probs_va[:,1]
        best_thr, best_score = tune_threshold(y_va, pos_prob_va, metric=args.tune_metric, steps=args.tune_steps)
        print(f"\n>>> Tuned threshold on validation ({args.tune_metric}) = {best_thr:.4f}  (score={best_score:.4f})")

        yhat_thr = (pos_prob_te >= best_thr).astype(int)
        metrics_thr, cm_thr, report_thr = metrics_from_preds(y_te, yhat_thr, pos_prob_te)

        print("\n=== Test Metrics (Tuned threshold) ===")
        for k,v in metrics_thr.items(): print(f"{k.replace('_',' ').title():18s}: {v:.4f}")
        print("\n=== Per-class PRF (Tuned) ===")
        for cls in range(2):
            prf = report_thr[str(cls)]
            print(f"class {cls}: precision={prf['precision']:.4f} recall={prf['recall']:.4f} f1={prf['f1-score']:.4f}")

        with open(os.path.join(run_dir, "tuned_threshold.txt"), "w") as f:
            f.write(f"{best_thr:.6f}\n")
        with open(os.path.join(run_dir, "test_metrics_tuned.json"), "w") as f:
            json.dump({"metrics":metrics_thr,"report":report_thr,"threshold":best_thr}, f, indent=2)
        plot_confusion_matrix(cm_thr, class_names, os.path.join(run_dir, "confusion_matrix_tuned.png"))
        tuned = {"threshold": best_thr, "metrics": metrics_thr, "report": report_thr}

    # Save PR/ROC for argmax (already) and tuned (same curves, different operating point)
    # Per-example CSVs
    out_csv_argmax = os.path.join(run_dir, "predictions_argmax.csv")
    with open(out_csv_argmax, "w") as f:
        f.write("true,pred,prob_0,prob_1,confidence\n")
        for i in range(len(y_te)):
            conf = max(probs_te[i,0], probs_te[i,1])
            f.write(f"{int(y_te[i])},{int(y_pred_te[i])},{probs_te[i,0]:.6f},{probs_te[i,1]:.6f},{float(conf):.6f}\n")

    if args.tune_threshold:
        out_csv_tuned = os.path.join(run_dir, "predictions_tuned.csv")
        best_thr = tuned["threshold"]
        with open(out_csv_tuned, "w") as f:
            f.write("true,pred,prob_0,prob_1,confidence,threshold\n")
            for i in range(len(y_te)):
                pred = int(pos_prob_te[i] >= best_thr)
                conf = max(probs_te[i,0], probs_te[i,1])
                f.write(f"{int(y_te[i])},{pred},{probs_te[i,0]:.6f},{probs_te[i,1]:.6f},{float(conf):.6f},{best_thr:.6f}\n")

    # Summary
    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write("TinyCNN with rotation+flip augmentation (train only)\n")
        f.write(f"Run dir: {run_dir}\n")
        f.write(f"Params: total={total_p} trainable={train_p}\n")
        f.write(f"Augment: rot=±{args.rot_degrees}°, hflip_p={args.flip_p_h}, vflip_p={args.flip_p_v}, interp={args.interp}, fill={0.0 if args.standardize else float(Xtr.mean().item())}\n")
        f.write(f"Label smoothing: {args.label_smoothing}\n")
        f.write("[Argmax metrics]\n")
        for k,v in metrics_argmax.items(): f.write(f"{k}: {v:.4f}\n")
        if args.tune_threshold:
            f.write("\n[Tuned metrics]\n")
            tm = tuned["metrics"]
            for k,v in tm.items(): f.write(f"{k}: {v:.4f}\n")
            f.write(f"threshold: {tuned['threshold']:.6f}\n")

    print("\nDone. Artifacts saved to:", run_dir)

if __name__ == "__main__":
    main()

