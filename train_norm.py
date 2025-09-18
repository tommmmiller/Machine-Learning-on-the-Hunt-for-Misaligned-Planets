import os
import numpy as np
import pandas as pd

# === Paths ===
TRAIN_DIR = os.path.abspath("train")
DIFF_DIR = os.path.join(TRAIN_DIR, "diff")
DIFF_NORM_DIR = os.path.join(TRAIN_DIR, "diff_norm")
META_PATH = os.path.join(TRAIN_DIR, "metadata.csv")

# === Create output directory ===
os.makedirs(DIFF_NORM_DIR, exist_ok=True)

# === Read metadata ===
meta_df = pd.read_csv(META_PATH)

# === Group by simulation name ===
sim_groups = meta_df.groupby("simulation")

for sim_name, group in sim_groups:
    print(f"Processing normalization for {sim_name}")

    # Load all associated diff files for this simulation
    diffs = []
    co_indices = []
    for _, row in group.iterrows():
        fname = row["diff_path"]
        fpath = os.path.join(DIFF_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  Missing file: {fpath}")
            continue
        arr = np.load(fpath)
        diffs.append(arr)
        co_indices.append(int(row["co_index"]))

    if not diffs:
        print(f"  No data found for {sim_name}")
        continue

    diffs = np.stack(diffs)
    max_val = np.max(np.abs(diffs))
    if max_val == 0:
        print(f"  Skipping {sim_name}, max diff is zero.")
        continue

    # Normalize to [-1, 1]
    normed = diffs / max_val

    for norm_arr, co_idx in zip(normed, co_indices):
        condensed_name = f"{sim_name}_diff_{co_idx}.npy"
        out_path = os.path.join(DIFF_NORM_DIR, condensed_name)
        np.save(out_path, norm_arr)
        print(f"  Saved normalized: {out_path}")

print("\nNormalization complete. All files saved to 'diff_norm'.")
