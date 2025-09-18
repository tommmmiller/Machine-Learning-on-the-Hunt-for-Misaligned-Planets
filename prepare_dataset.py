import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# === Paths ===
BASE_DIR = os.path.abspath("train")
DIFF_NORM_DIR = os.path.join(BASE_DIR, "diff_norm")
META_PATH = os.path.join(BASE_DIR, "metadata.csv")

# === Load metadata ===
meta_df = pd.read_csv(META_PATH)

# Filter to only include files that were normalized
diff_files = set(os.listdir(DIFF_NORM_DIR))
meta_df["condensed_name"] = meta_df.apply(
    lambda row: f"{row['simulation']}_diff_{int(row['co_index'])}.npy", axis=1
)
meta_df = meta_df[meta_df["condensed_name"].isin(diff_files)]

# Assign binary labels: 1 = misaligned (inc ≥ 3), 0 = coplanar (inc < 3)
meta_df["label"] = meta_df["simulation"].apply(
    lambda s: 1 if float(s.split("_")[1].replace("inc", "")) >= 3 else 0
)

# Extract inclination for analysis
meta_df["incl"] = meta_df["simulation"].apply(
    lambda s: float(s.split("_")[1].replace("inc", ""))
)

# === Load all data ===
X = []
y = []
incl = []

for _, row in meta_df.iterrows():
    path = os.path.join(DIFF_NORM_DIR, row["condensed_name"])
    arr = np.load(path).astype(np.float32)
    if arr.shape != (301, 301):
        print(f"Skipping {path}, wrong shape: {arr.shape}")
        continue
    X.append(arr)
    y.append(row["label"])
    incl.append(row["incl"])

X = np.stack(X)       # shape: (N, 301, 301)
y = np.array(y)       # shape: (N,)
incl = np.array(incl) # shape: (N,)

# Add channel dimension for PyTorch CNN
X = X[:, None, :, :]  # shape: (N, 1, 301, 301)

# === Train/test split ===
X_train, X_test, y_train, y_test, incl_train, incl_test = train_test_split(
    X, y, incl, test_size=0.2, stratify=y, random_state=42
)

# === Save to disk as .pt ===
torch.save(torch.tensor(X_train), os.path.join(BASE_DIR, "X_train.pt"))
torch.save(torch.tensor(y_train), os.path.join(BASE_DIR, "y_train.pt"))
torch.save(torch.tensor(X_test), os.path.join(BASE_DIR, "X_test.pt"))
torch.save(torch.tensor(y_test), os.path.join(BASE_DIR, "y_test.pt"))
np.save(os.path.join(BASE_DIR, "incl_test.npy"), incl_test)

print("Dataset preparation complete.")
print(f"X_train shape: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test: {y_test.shape}")
