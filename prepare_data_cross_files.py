import os
import re
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

# ----------------- Filename parsing -----------------

# accepts: simXX_..._raw_15.npy  OR  simXX_..._raw_15_idx7.npy
SIM_RE = re.compile(r"^(?P<sim>.+?)_raw_\d+(?:_idx\d+)?\.npy$")

INCL_RE = re.compile(r"inc(?P<incl>\d+(?:\.\d+)?)")

def parse_sim_and_incl(filename: str):
    """
    Given e.g. 'sim50_mp4.55_inc5.66_a77.1_phi30.73_raw_24_idx5.npy',
    returns (simulation_str, inclination_float).
    simulation_str = everything before '_raw_...'
    inclination = float parsed from 'inc<value>'.
    """
    m_sim = SIM_RE.match(filename)
    if not m_sim:
        raise ValueError(f"Filename not recognized: {filename}")
    sim = m_sim.group("sim")
    m_incl = INCL_RE.search(sim)
    if not m_incl:
        raise ValueError(f"Could not find 'inc<val>' in simulation name: {sim}")
    incl = float(m_incl.group("incl"))
    return sim, incl

def bin_label_from_incl(incl: float) -> int:
    # Your original binning: [0,9) -> 0, [9,25) -> 1
    if 0 <= incl < 9:
        return 0
    elif 9 <= incl < 25:
        return 1
    else:
        raise ValueError(f"Inclination {incl} outside expected range [0,25)")

# ----------------- Loader from directory -----------------

def list_npy(dirpath: str):
    return sorted([f for f in os.listdir(dirpath) if f.endswith(".npy")])

def build_index_from_dir(dirpath: str):
    """
    Scans a 'raw' directory and returns a DataFrame with:
    ['filename', 'simulation', 'incl', 'label']
    """
    rows = []
    for fn in list_npy(dirpath):
        try:
            sim, incl = parse_sim_and_incl(fn)
            label = bin_label_from_incl(incl)
            rows.append((fn, sim, incl, label))
        except Exception as e:
            print(f"[WARN] Skipping {fn}: {e}")
    if not rows:
        raise RuntimeError(f"No valid .npy files parsed in {dirpath}")
    df = pd.DataFrame(rows, columns=["filename", "simulation", "incl", "label"])
    return df

def load_stack(df: pd.DataFrame, data_dir: str):
    X, y, incl = [], [], []
    for _, r in df.iterrows():
        path = os.path.join(data_dir, r["filename"])
        arr = np.load(path).astype(np.float32)
        if arr.shape != (301, 301):
            print(f"[WARN] Skipping {r['filename']} shape={arr.shape}")
            continue
        X.append(arr); y.append(int(r["label"])); incl.append(float(r["incl"]))
    if len(X) == 0:
        raise RuntimeError(f"No arrays loaded from {data_dir}")
    X = np.stack(X)[:, None, :, :]
    y = np.asarray(y, dtype=np.int64)
    incl = np.asarray(incl, dtype=np.float32)
    return X, y, incl

# ----------------- Main -----------------

def main():
    ap = argparse.ArgumentParser(description="Prepare train/val/test tensors from filenames (no metadata.csv).")
    ap.add_argument("--train_src_base", type=str, required=True,
                    help="Source folder for TRAIN (e.g., 'train_3' or 'train_all'). Uses <base>/raw/*.npy")
    ap.add_argument("--valtest_base", type=str, default="train",
                    help="Folder for VAL/TEST (original). Uses <base>/raw/*.npy and <base>/heldout_sims.csv")
    ap.add_argument("--valtest_model_dir", type=str, default="train/raw/model",
                    help="Directory that may contain split_by_simulation.csv to REUSE val sims.")
    ap.add_argument("--out_model_dir", type=str, required=True,
                    help="Where to save X_tr.pt, X_val.pt, X_test.pt etc.")
    ap.add_argument("--val_frac", type=float, default=0.2,
                    help="Val fraction if we need to create a val split.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dry_run", action="store_true", help="Print counts and exit without writing files.")
    args = ap.parse_args()

    # Directories
    train_src_raw = os.path.join(args.train_src_base, "raw")
    valtest_raw   = os.path.join(args.valtest_base, "raw")
    heldout_csv   = os.path.join(args.valtest_base, "heldout_sims.csv")

    # Preconditions
    if not os.path.isdir(train_src_raw):
        raise FileNotFoundError(f"Missing train source raw dir: {train_src_raw}")
    if not os.path.isdir(valtest_raw):
        raise FileNotFoundError(f"Missing val/test raw dir: {valtest_raw}")
    if not os.path.exists(heldout_csv):
        raise FileNotFoundError(f"Missing heldout_sims.csv at {heldout_csv}")

    # Build indices from filenames only
    train_src_df = build_index_from_dir(train_src_raw)
    valtest_df   = build_index_from_dir(valtest_raw)

    # Test simulations from heldout_sims.csv
    heldout = pd.read_csv(heldout_csv)
    if "simulation" not in heldout.columns:
        raise KeyError(f"{heldout_csv} must contain a 'simulation' column")
    test_sims = set(heldout["simulation"].astype(str))

    # Validation sims: reuse if available
    split_csv = os.path.join(args.valtest_model_dir, "split_by_simulation.csv")
    if os.path.exists(split_csv):
        split_map = pd.read_csv(split_csv)
        if not {"simulation","split"} <= set(split_map.columns):
            raise KeyError(f"{split_csv} must contain 'simulation' and 'split' columns")
        val_sims = set(split_map.loc[split_map["split"]=="val","simulation"].astype(str))
        if not val_sims:
            raise RuntimeError(f"No 'val' sims found in {split_csv}")
        reused_val = True
    else:
        # Create a split by simulation on the original (non-heldout) pool
        pool_df = valtest_df[~valtest_df["simulation"].isin(test_sims)].copy()
        groups = pool_df["simulation"].to_numpy()
        labels = pool_df["label"].to_numpy()
        gss = GroupShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed)
        _, val_idx = next(gss.split(np.zeros(len(labels)), labels, groups))
        val_sims = set(pool_df.iloc[val_idx]["simulation"].astype(str))
        reused_val = False

    # Build TEST from original held-out sims
    test_df = valtest_df[valtest_df["simulation"].isin(test_sims)].copy()
    # Build VAL from original val sims
    val_df = valtest_df[valtest_df["simulation"].isin(val_sims)].copy()

    # TRAIN from train_src, excluding any sim used in val/test
    forbidden = val_sims.union(test_sims)
    train_df = train_src_df[~train_src_df["simulation"].isin(forbidden)].copy()
    if train_df.empty:
        raise RuntimeError("Train set empty after excluding val/test sims. Check sim name consistency across sources.")

    # Dry run summary
    print("\n=== Plan (dry-run counts) ===")
    print(f"Train source: {args.train_src_base} | files parsed: {len(train_src_df)} | sims: {train_src_df['simulation'].nunique()}")
    print(f"Val/Test src: {args.valtest_base}   | files parsed: {len(valtest_df)} | sims: {valtest_df['simulation'].nunique()}")
    print(f"Val sims: {len(val_sims)} ({'reused' if reused_val else 'new split'}) | Test sims: {len(test_sims)}")
    print(f"TRAIN files selected: {len(train_df)} | VAL: {len(val_df)} | TEST: {len(test_df)}")
    if args.dry_run:
        return

    # Load arrays
    X_tr, y_tr, _ = load_stack(train_df, train_src_raw)
    X_val, y_val, _ = load_stack(val_df,   valtest_raw)
    X_te, y_te, incl_te = load_stack(test_df,  valtest_raw)

    # Create output dir
    os.makedirs(args.out_model_dir, exist_ok=True)

    # Save tensors
    torch.save(torch.tensor(X_tr),                 os.path.join(args.out_model_dir, "X_tr.pt"))
    torch.save(torch.tensor(y_tr, dtype=torch.long), os.path.join(args.out_model_dir, "y_tr.pt"))
    torch.save(torch.tensor(X_val),                os.path.join(args.out_model_dir, "X_val.pt"))
    torch.save(torch.tensor(y_val, dtype=torch.long), os.path.join(args.out_model_dir, "y_val.pt"))
    torch.save(torch.tensor(X_te),                 os.path.join(args.out_model_dir, "X_test.pt"))
    torch.save(torch.tensor(y_te, dtype=torch.long), os.path.join(args.out_model_dir, "y_test.pt"))
    np.save(os.path.join(args.out_model_dir, "incl_test.npy"), incl_te)

    # Save bookkeeping
    pd.DataFrame({"simulation": sorted(train_df["simulation"].unique())}).to_csv(
        os.path.join(args.out_model_dir, "train_sims.csv"), index=False
    )
    pd.DataFrame({"simulation": sorted(val_df["simulation"].unique())}).to_csv(
        os.path.join(args.out_model_dir, "val_sims.csv"), index=False
    )
    pd.DataFrame({"simulation": sorted(test_df["simulation"].unique())}).to_csv(
        os.path.join(args.out_model_dir, "test_sims.csv"), index=False
    )

    # If we created a new val split, save it for future reuse
    if not reused_val:
        os.makedirs(args.valtest_model_dir, exist_ok=True)
        map_df = valtest_df.copy()
        map_df["split"] = np.where(map_df["simulation"].isin(val_sims), "val", "train")
        map_df[["simulation","split"]].drop_duplicates().to_csv(split_csv, index=False)
        print(f"Saved new validation split map to {split_csv}")

    # Report
    print("\n=== Saved ===")
    print(f"Out dir: {args.out_model_dir}")
    print(f"Train: X={X_tr.shape}, labels={np.bincount(y_tr, minlength=2)}")
    print(f"Val:   X={X_val.shape}, labels={np.bincount(y_val, minlength=2)}")
    print(f"Test:  X={X_te.shape}, labels={np.bincount(y_te, minlength=2)}")
    print("Files: X_tr.pt, y_tr.pt, X_val.pt, y_val.pt, X_test.pt, y_test.pt, incl_test.npy, "
          "train_sims.csv, val_sims.csv, test_sims.csv")

if __name__ == "__main__":
    main()

