import os
import numpy as np
import pandas as pd
import argparse
import torch
from sklearn.model_selection import GroupShuffleSplit

# ---------- Helpers ----------

def extract_incl(sim_name: str) -> float:
    parts = [p for p in sim_name.split("_") if p.startswith("inc")]
    if not parts:
        raise ValueError(f"No 'inc...' part found in simulation string: {sim_name}")
    return float(parts[0].replace("inc", ""))

def inclination_to_bin(incl: float) -> int:
    if 0 <= incl < 9:
        return 0
    elif 9 <= incl < 25:
        return 1
    else:
        raise ValueError(f"Inclination {incl} outside expected range [0,25)")

def load_meta_filtered(base_dir: str, subdir: str = "raw", filename_col: str = "raw_path"):
    """Load metadata.csv from base_dir, attach filename (from filename_col),
    filter to files that exist under base_dir/subdir, and add incl/label columns.
    Returns (df, data_dir)."""
    meta_path = os.path.join(base_dir, "metadata.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}")
    df = pd.read_csv(meta_path)
    if filename_col not in df.columns:
        raise KeyError(f"{meta_path} missing column '{filename_col}'. Available: {list(df.columns)}")

    data_dir = os.path.join(base_dir, subdir)
    df["filename"] = df[filename_col].astype(str)
    existing = set(os.listdir(data_dir))
    df = df[df["filename"].isin(existing)].copy()
    if df.empty:
        raise RuntimeError(f"No files from metadata exist under {data_dir}.")

    if "incl" not in df.columns:
        df["incl"] = df["simulation"].apply(extract_incl)
    if "label" not in df.columns:
        df["label"] = df["incl"].apply(inclination_to_bin)

    return df, data_dir

def load_stack(df: pd.DataFrame, data_dir: str):
    X, y, incl = [], [], []
    for _, row in df.iterrows():
        path = os.path.join(data_dir, row["filename"])
        arr = np.load(path).astype(np.float32)
        if arr.shape != (301, 301):
            # drop silently but notify; you can hard-error if preferred
            print(f"Skipping {path}, unexpected shape {arr.shape}")
            continue
        X.append(arr); y.append(int(row["label"])); incl.append(float(row["incl"]))
    if len(X) == 0:
        raise RuntimeError(f"No arrays loaded from {data_dir} (after shape filtering).")
    X = np.stack(X)[:, None, :, :]
    y = np.asarray(y, dtype=np.int64)
    incl = np.asarray(incl, dtype=np.float32)
    return X, y, incl

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Train on train_src_base, val/test from valtest_base.")
    ap.add_argument("--train_src_base", type=str, default="train_3",
                    help="Folder with training source (e.g., train_3). Uses <base>/metadata.csv and <base>/raw.")
    ap.add_argument("--valtest_base", type=str, default="train",
                    help="Folder with validation/test source (original). Uses <base>/metadata.csv, <base>/raw and heldout_sims.csv.")
    ap.add_argument("--valtest_model_dir", type=str, default="train/raw/model",
                    help="Existing model dir that may contain split_by_simulation.csv to REUSE val sims.")
    ap.add_argument("--out_model_dir", type=str, default="train/raw/model_3best_to_train",
                    help="Where to save tensors for training with train_src and val/test from valtest.")
    ap.add_argument("--val_frac", type=float, default=0.2, help="Val fraction if we need to create a split.")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    # Paths
    os.makedirs(args.out_model_dir, exist_ok=True)

    # --- Load metadata for train source (e.g., train_3/raw) ---
    train_src_df, train_src_data_dir = load_meta_filtered(args.train_src_base, subdir="raw", filename_col="raw_path")

    # --- Load metadata for val/test source (original train/raw) ---
    valtest_df, valtest_data_dir = load_meta_filtered(args.valtest_base, subdir="raw", filename_col="raw_path")

    # --- Held-out simulations (test set) from original ---
    heldout_path = os.path.join(args.valtest_base, "heldout_sims.csv")
    if not os.path.exists(heldout_path):
        raise FileNotFoundError(f"Missing heldout_sims.csv at {heldout_path}")
    heldout_sims = pd.read_csv(heldout_path)
    if "simulation" not in heldout_sims.columns:
        raise KeyError(f"{heldout_path} must have a 'simulation' column.")
    test_sims = set(heldout_sims["simulation"].astype(str))

    # --- Determine validation simulations (reuse if available) ---
    split_map_path = os.path.join(args.valtest_model_dir, "split_by_simulation.csv")
    if os.path.exists(split_map_path):
        split_map = pd.read_csv(split_map_path)
        if "simulation" not in split_map.columns or "split" not in split_map.columns:
            raise KeyError(f"{split_map_path} must contain 'simulation' and 'split' columns.")
        val_sims = set(split_map.loc[split_map["split"] == "val", "simulation"].astype(str))
        if not val_sims:
            raise RuntimeError(f"No 'val' sims found in {split_map_path}.")
        print(f"Reusing validation simulations from {split_map_path}: {len(val_sims)} sims.")
    else:
        # Create a split by simulation from the original (non-heldout) pool
        pool_df = valtest_df[~valtest_df["simulation"].isin(test_sims)].copy()
        groups = pool_df["simulation"].to_numpy()
        labels = pool_df["label"].to_numpy()
        gss = GroupShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed)
        _, val_idx = next(gss.split(np.zeros(len(labels)), labels, groups))
        val_sims = set(pool_df.iloc[val_idx]["simulation"].astype(str))
        print(f"Created new validation split by simulation ({len(val_sims)} sims).")
        # Save split map for reproducibility (mark only val; train sims are implicit)
        map_df = pool_df[["simulation","label"]].copy()
        map_df["split"] = np.where(pool_df["simulation"].isin(val_sims), "val", "train")
        map_df = map_df.drop_duplicates(subset=["simulation","split"])
        os.makedirs(args.valtest_model_dir, exist_ok=True)
        map_df.to_csv(split_map_path, index=False)
        print(f"Saved {split_map_path}")

    # --- Build TEST set from original source & held-out sims ---
    test_df = valtest_df[valtest_df["simulation"].isin(test_sims)].copy()
    X_te, y_te, incl_te = load_stack(test_df, valtest_data_dir)

    # --- Build VAL set from original source & val sims ---
    val_df = valtest_df[valtest_df["simulation"].isin(val_sims)].copy()
    X_val, y_val, _ = load_stack(val_df, valtest_data_dir)

    # --- Build TRAIN set from train_src source, excluding val+test sims to avoid leakage ---
    forbidden_sims = val_sims.union(test_sims)
    train_df = train_src_df[~train_src_df["simulation"].isin(forbidden_sims)].copy()
    if train_df.empty:
        raise RuntimeError("Train set is empty after excluding val/test sims. Check sim name consistency across sources.")
    X_tr, y_tr, _ = load_stack(train_df, train_src_data_dir)

    # --- Save tensors ---
    torch.save(torch.tensor(X_tr),  os.path.join(args.out_model_dir, "X_tr.pt"))
    torch.save(torch.tensor(y_tr, dtype=torch.long),  os.path.join(args.out_model_dir, "y_tr.pt"))
    torch.save(torch.tensor(X_val), os.path.join(args.out_model_dir, "X_val.pt"))
    torch.save(torch.tensor(y_val, dtype=torch.long), os.path.join(args.out_model_dir, "y_val.pt"))
    torch.save(torch.tensor(X_te),  os.path.join(args.out_model_dir, "X_test.pt"))
    torch.save(torch.tensor(y_te, dtype=torch.long),  os.path.join(args.out_model_dir, "y_test.pt"))
    np.save(os.path.join(args.out_model_dir, "incl_test.npy"), incl_te)

    # --- Save bookkeeping (useful for audits) ---
    pd.DataFrame({"simulation": sorted(list(val_sims))}).to_csv(os.path.join(args.out_model_dir, "val_sims.csv"), index=False)
    pd.DataFrame({"simulation": sorted(list(test_sims))}).to_csv(os.path.join(args.out_model_dir, "test_sims.csv"), index=False)
    pd.DataFrame({"simulation": sorted(list(set(train_df['simulation'].astype(str))))}).to_csv(os.path.join(args.out_model_dir, "train_sims.csv"), index=False)

    # --- Report ---
    print("\n=== Cross-source dataset summary ===")
    print(f"Train source dir:     {train_src_data_dir}")
    print(f"Val/Test source dir:  {valtest_data_dir}")
    print(f"Train: X={X_tr.shape}, labels={np.bincount(y_tr, minlength=2)} (unique sims: {len(set(train_df['simulation']))})")
    print(f"Val:   X={X_val.shape}, labels={np.bincount(y_val, minlength=2)} (unique sims: {len(val_sims)})")
    print(f"Test:  X={X_te.shape}, labels={np.bincount(y_te, minlength=2)} (unique sims: {len(test_sims)})")
    print(f"Saved tensors and sim lists to: {args.out_model_dir}")

if __name__ == "__main__":
    main()

