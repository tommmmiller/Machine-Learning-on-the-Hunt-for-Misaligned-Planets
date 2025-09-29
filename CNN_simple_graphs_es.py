import os, json, time, argparse, math, random
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, average_precision_score, roc_auc_score,
                             precision_recall_curve, roc_curve, confusion_matrix,
                             classification_report)
from sklearn.calibration import calibration_curve
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt

# ------------------------- Config / Repro ------------------------------------
def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def parse_grid(spec: str):
    # "wd=0,1e-4;lr=1e-3,1e-4"
    parts = [p.strip() for p in spec.split(";") if p.strip()]
    grid = {}
    for p in parts:
        k, vs = p.split("=")
        grid[k.strip()] = [float(x) for x in vs.split(",")]
    return grid
def train_one_logreg(model, train_loader, val_loader, device, lr, weight_decay, epochs=50, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_metric, wait = -1.0, 0
    best_state = None

    for epoch in range(1, epochs+1):
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
            total_loss += loss.item()

        # val
        _, _, _, _, val_metrics, _, _ = evaluate(model, val_loader, device)
        val_f1 = val_metrics["macro_f1"]
        scheduler.step(val_f1)

        if val_f1 > best_metric + 1e-6:
            best_metric, wait = val_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    # restore best
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return best_metric

# ------------------------- Model ---------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,16,5,padding=2), nn.ReLU(), nn.MaxPool2d(2),   # 150x150 -> 75x75 after two pools
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*75*75, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    def forward(self,x): return self.head(self.conv(x))

class TinyCNN(nn.Module):
    """
    ~25k params. Two downsamplings (301→150→75), light conv stack, GAP head.
    BN helps optimization; GAP kills huge FC layers = less overfitting.
    """
    def __init__(self, in_ch=1, num_classes=2, p_drop=0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.AvgPool2d(2),                       # 301 -> 150

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.AvgPool2d(2),                       # 150 -> 75

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 64 x 1 x 1
            nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)
class Standardizer(nn.Module):
    def __init__(self, in_shape):
        super().__init__()
        c,h,w = in_shape
        self.register_buffer("mean", torch.zeros(1, c, h, w))
        self.register_buffer("std",  torch.ones(1, c, h, w))

    @torch.no_grad()
    def fit(self, x):  # x: (N,C,H,W) train tensor
        m = x.mean(dim=0, keepdim=True)
        s = x.std(dim=0, keepdim=True).clamp_min(1e-6)
        self.mean.copy_(m)
        self.std.copy_(s)

    def forward(self, x):
        return (x - self.mean) / self.std

class LogisticRegression(nn.Module):
    """Multinomial logistic regression (softmax) on flattened pixels."""
    def __init__(self, in_shape, n_classes=2):
        super().__init__()
        c,h,w = in_shape
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(c*h*w, n_classes)  # no hidden layers
    def forward(self, x):
        return self.fc(self.flatten(x))

def count_params(model):
    return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)

# ------------------------- Plot helpers (matplotlib only) --------------------
def plot_history(run_dir, hist):
    # hist: list of dicts with keys 'epoch','train_loss','val_acc','val_f1'
    es = [h['epoch'] for h in hist]
    tl = [h['train_loss'] for h in hist]
    vf = [h['val_f1'] for h in hist]

    plt.figure()
    plt.plot(es, tl); plt.xlabel("Epoch"); plt.ylabel("Train loss"); plt.title("Training loss")
    plt.tight_layout(); plt.savefig(os.path.join(run_dir, "train_loss.png")); plt.close()

    plt.figure()
    plt.plot(es, vf); plt.xlabel("Epoch"); plt.ylabel("Val Macro-F1"); plt.title("Validation Macro-F1")
    plt.tight_layout(); plt.savefig(os.path.join(run_dir, "val_f1.png")); plt.close()

def plot_confusion_matrix(cm, class_names, out_path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label', title=title)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path); plt.close(fig)

def plot_pr_roc(y_true, y_score, run_dir):
    # Precision-Recall
    p, r, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(r, p)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve")
    plt.tight_layout(); plt.savefig(os.path.join(run_dir, "pr_curve.png")); plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
    plt.tight_layout(); plt.savefig(os.path.join(run_dir, "roc_curve.png")); plt.close()

def plot_reliability(y_true, y_score, run_dir, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy='uniform')
    plt.figure()
    plt.plot([0,1],[0,1],'--')
    plt.plot(prob_pred, prob_true, marker='o')
    plt.xlabel("Predicted probability"); plt.ylabel("Empirical probability")
    plt.title("Reliability Diagram")
    plt.tight_layout(); plt.savefig(os.path.join(run_dir, "reliability.png")); plt.close()

def plot_confidence_hist(confidences, out_path):
    plt.figure()
    plt.hist(confidences, bins=20, alpha=0.8)
    plt.xlabel("Prediction confidence"); plt.ylabel("Count"); plt.title("Confidence Distribution")
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_confidence_vs_incl(incl, conf, correct, out_path):
    plt.figure(figsize=(7,5))
    sc = plt.scatter(incl, conf, c=correct.astype(int), cmap="bwr", alpha=0.6)
    cbar = plt.colorbar(sc); cbar.set_label("Correct (1) / Incorrect (0)")
    plt.xlabel("Inclination"); plt.ylabel("Prediction confidence")
    plt.title("Confidence vs Inclination")
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_avg_probs_vs_incl(incl, probs, out_path):
    # incl: (N,), probs: (N,2) for binary
    order = np.argsort(incl)
    incl_sorted = incl[order]
    probs_sorted = probs[order]
    # simple binning by unique incl if discrete, else 20 bins
    uniq = np.unique(incl_sorted)
    if len(uniq) <= 25:
        xs = uniq
        avg = np.vstack([probs_sorted[incl_sorted==u].mean(0) for u in uniq])
    else:
        bins = np.linspace(incl_sorted.min(), incl_sorted.max(), 21)
        idx = np.digitize(incl_sorted, bins)-1
        xs = 0.5*(bins[:-1]+bins[1:])
        avg = np.vstack([probs_sorted[idx==b].mean(0) if np.any(idx==b) else np.full(2, np.nan) for b in range(len(xs))])
    plt.figure(figsize=(7,5))
    for k in range(probs.shape[1]):
        plt.plot(xs, avg[:,k], label=f"Class {k}")
    plt.xlabel("Inclination"); plt.ylabel("Avg predicted probability")
    plt.title("Predicted Probabilities vs Inclination")
    plt.legend(); plt.tight_layout(); plt.savefig(out_path); plt.close()

# ------------------------- Train / Eval --------------------------------------
def evaluate(model, loader, device):
    model.eval()
    ys, ps, logits_all, confs = [], [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            lg = model(xb)
            pr = torch.softmax(lg, dim=1)
            pred = pr.argmax(dim=1).cpu().numpy()
            conf = pr.max(dim=1).values.cpu().numpy()
            ys.append(yb.numpy()); ps.append(pred); logits_all.append(pr.cpu().numpy()); confs.append(conf)
    y = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    probs = np.concatenate(logits_all)                    # (N,2)
    confs = np.concatenate(confs)                         # (N,)
    pos_prob = probs[:,1]                                 # assume class 1 is positive
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "macro_f1": f1_score(y, y_pred, average="macro"),
        "mcc": matthews_corrcoef(y, y_pred),
        "auc_pr": average_precision_score(y, pos_prob),
        "roc_auc": roc_auc_score(y, pos_prob)
    }
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    return y, y_pred, probs, confs, metrics, cm, report
import argparse
import numpy as np

def make_random_labels(y_np, mode="off", seed=1234, prior_from=None, n_classes=2):
    """
    y_np: numpy array of true labels
    mode: "off" | "permute" | "iid-empirical" | "iid-uniform"
    prior_from: numpy array to compute empirical prior from (usually train labels)
    """
    rng = np.random.default_rng(seed)
    y_np = np.asarray(y_np)

    if mode == "off":
        return y_np

    if mode == "permute":
        return rng.permutation(y_np)

    if mode == "iid-empirical":
        ref = y_np if prior_from is None else np.asarray(prior_from)
        counts = np.bincount(ref, minlength=n_classes)
        p = counts / counts.sum()
        return rng.choice(n_classes, size=len(y_np), p=p)

    if mode == "iid-uniform":
        return rng.integers(low=0, high=n_classes, size=len(y_np))

    raise ValueError(f"Unknown random label mode: {mode}")

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # I/O
    base = os.getcwd()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base, "runs", f"simplecnn_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    # Data
    X_train = torch.load(os.path.join(base,"X_tr.pt")).float()
    y_train = torch.load(os.path.join(base,"y_tr.pt")).long()
    X_val   = torch.load(os.path.join(base,"X_val.pt")).float()
    y_val   = torch.load(os.path.join(base,"y_val.pt")).long()
    X_test  = torch.load(os.path.join(base,"X_test.pt")).float()
    y_test  = torch.load(os.path.join(base,"y_test.pt")).long()
    incl_te = np.load(os.path.join(base,"incl_test.npy"))

    # y_tr, y_val, y_test are torch tensors (long). Convert to numpy for shuffling:
    y_tr_np  = y_train.cpu().numpy()
    y_val_np = y_val.cpu().numpy()

    # For iid-empirical, use TRAIN prior for both train and val:
    y_tr_rand  = make_random_labels(y_tr_np,
                                mode=args.random_labels,
                                seed=args.random_label_seed,
                                prior_from=y_tr_np,
                                n_classes=2)
    y_val_rand = make_random_labels(y_val_np,
                                mode=args.random_labels,
                                seed=args.random_label_seed + 1,  # different seed for val
                                prior_from=y_tr_np,
                                n_classes=2)

    # Put back into tensors (train/val randomized; TEST remains TRUE labels)
    y_train  = torch.tensor(y_tr_rand, dtype=torch.long, device=y_train.device)
    y_val = torch.tensor(y_val_rand, dtype=torch.long, device=y_val.device)

    # Standardize using TRAIN statistics (channel-wise)
    if args.standardize:
        with torch.no_grad():
            mu = X_train.mean(dim=(0,2,3), keepdim=True)
            sd = X_train.std(dim=(0,2,3), keepdim=True).clamp_min(1e-6)
            X_train = (X_train - mu)/sd
            X_val   = (X_val   - mu)/sd
            X_test  = (X_test  - mu)/sd


    train_loader = DataLoader(TensorDataset(X_train,y_train), batch_size=args.bs, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(TensorDataset(X_val,y_val),     batch_size=2*args.bs, shuffle=False, pin_memory=True)
    test_loader  = DataLoader(TensorDataset(X_test,y_test),   batch_size=2*args.bs, shuffle=False, pin_memory=True)

    # Model / loss / opt
    in_shape = tuple(X_train.shape[1:])  # (C,H,W) e.g. (1,301,301)

    if args.model == "cnn":
        model = SimpleCNN().to(device)
    elif args.model == "logreg":
        model = LogisticRegression().to(device)
    elif args.model == "tinycnn":
        model = TinyCNN().to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    total_p, train_p = count_params(model)

   
    counts = np.bincount(y_train.numpy(), minlength=2).astype(np.float32)
    inv = 1.0 / np.maximum(counts, 1)
    w = (inv / inv.sum()) * 2.0
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Training
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best_metric, patience, wait = -1.0, args.patience, 0
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
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

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
            if wait >= patience:
                print("Early stopping.")
                break

    # Save training artifacts
    with open(os.path.join(run_dir, "history.json"), "w") as f: json.dump(history, f, indent=2)
    plot_history(run_dir, history)

    # Reload best and evaluate on test
    model.load_state_dict(torch.load(best_path, map_location=device))
    y, y_pred, probs, confs, metrics, cm, report = evaluate(model, test_loader, device)
    pos_prob = probs[:,1]
    class_names = [f"Class {i}" for i in range(2)]

    # Print metrics
    print("\n=== Test Metrics ===")
    for k,v in metrics.items():
        print(f"{k.replace('_',' ').title():18s}: {v:.4f}")
    print("\n=== Per-class PRF ===")
    for cls in range(2):
        prf = report[str(cls)]
        print(f"class {cls}: precision={prf['precision']:.4f} recall={prf['recall']:.4f} f1={prf['f1-score']:.4f}")
    print(f"\nParams: total={total_p:,}  trainable={train_p:,}")

    # Save metrics + report
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f: json.dump({"metrics":metrics,"report":report,"params":{"total":int(total_p),"trainable":int(train_p)}}, f, indent=2)

    # Plots
    plot_confusion_matrix(cm, class_names, os.path.join(run_dir, "confusion_matrix.png"))
    plot_pr_roc(y, pos_prob, run_dir)
    plot_reliability(y, pos_prob, run_dir)
    plot_confidence_hist(confs, os.path.join(run_dir, "confidence_hist.png"))

    # Confidence vs inclination & avg probs vs inclination (if shapes match)
    if len(incl_te) == len(y):
        correct = (y_pred == y).astype(np.int32)
        plot_confidence_vs_incl(incl_te, confs, correct, os.path.join(run_dir, "confidence_vs_incl.png"))
        plot_avg_probs_vs_incl(incl_te, probs, os.path.join(run_dir, "avg_probs_vs_incl.png"))

    # Save per-example CSV
    out_csv = os.path.join(run_dir, "predictions.csv")
    header = "true,pred,prob_0,prob_1,confidence"
    if len(incl_te) == len(y): header += ",inclination"
    with open(out_csv, "w") as f:
        f.write(header+"\n")
        for i in range(len(y)):
            row = [int(y[i]), int(y_pred[i]), probs[i,0], probs[i,1], confs[i]]
            if len(incl_te) == len(y): row.append(float(incl_te[i]))
            f.write(",".join(map(str,row))+"\n")

    if args.model =="cnn":

        # Grad-CAMs
        print(f"\nGenerating {args.n_cam} Grad-CAMs...")
        target_layers = [model.conv[-3]]  # last Conv2d
        cam = GradCAM(model=model, target_layers=target_layers)

        # choose a mix of correct/incorrect if possible
        idx_all = np.arange(len(y))
        corr_idx = idx_all[y==y_pred]
        inc_idx  = idx_all[y!=y_pred]
        chosen = []
        if len(inc_idx) > 0:
            k = min(args.n_cam//2, len(inc_idx))
            chosen.extend(np.random.choice(inc_idx, k, replace=False).tolist())
        r = args.n_cam - len(chosen)
        if r > 0:
            pool = corr_idx if len(corr_idx) > 0 else idx_all
            chosen.extend(np.random.choice(pool, r, replace=False).tolist())

    # Need original tensors aligned with test_loader order
    # Reconstruct test tensors to index consistently
        X_test_full = X_test  # still in RAM
        for idx in chosen:
            input_tensor = X_test_full[idx:idx+1].to(device)
            grayscale_cam = cam(input_tensor=input_tensor)[0, :]
            img = X_test_full[idx,0].cpu().numpy()
            img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
            overlay = show_cam_on_image(np.stack([img_norm]*3, -1), grayscale_cam, use_rgb=True)
            plt.imshow(overlay); plt.axis("off")
            title = f"idx{idx}  pred={y_pred[idx]}  true={y[idx]}"
            if len(incl_te) == len(y): title += f"  incl={incl_te[idx]:.2f}"
            plt.title(title)
            out = os.path.join(run_dir, f"gradcam_{idx}.png")
            plt.savefig(out, bbox_inches="tight"); plt.close()
            print("Saved", out)
    else:
        print("Skipping Grad-CAM: not applicable for non-convolutional model.")

    # Short summary file
    summary_txt = os.path.join(run_dir, "summary.txt")
    with open(summary_txt, "w") as f:
        f.write("Test Summary\n")
        f.write(f"Run dir: {run_dir}\n")
        f.write(f"Model: {args.model}")
        for k,v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write(f"params_total: {total_p}\nparams_trainable: {train_p}\n")
        f.write("Artifacts: confusion_matrix.png, pr_curve.png, roc_curve.png, reliability.png, confidence_hist.png,\n")
        f.write("           confidence_vs_incl.png, avg_probs_vs_incl.png, predictions.csv, history.json\n")

    print("\nDone. Artifacts saved to:", run_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--n_cam", type=int, default=6)
    parser.add_argument("--random_labels", type=str, default="off",
                    choices=["off", "permute", "iid-empirical", "iid-uniform"],
                    help="Replace TRAIN and VAL labels with random labels for a null baseline.")
    parser.add_argument("--random_label_seed", type=int, default=1234,
                    help="Seed for random label generation.")
    parser.add_argument("--model", type=str, default="cnn",
                    choices=["cnn", "tinycnn", "logreg"],
                    help="Architecture: 'cnn' (SimpleCNN) or 'logreg' (logistic regression).")
    parser.add_argument("--standardize", action="store_true",
                    help="Z-score inputs using train mean/std before training and evaluation.")

    parser.add_argument("--logreg_grid", type=str,
        default="wd=0,1e-5,1e-4,1e-3;lr=1e-3,5e-4,1e-4",
        help="Grid over weight_decay (L2) and learning rates for logistic regression.")

    args = parser.parse_args()
    main(args)

