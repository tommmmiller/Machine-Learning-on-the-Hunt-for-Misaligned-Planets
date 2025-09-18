import os
import pandas as pd

# === Paths ===
TRAIN_DIR = os.path.abspath("train")
META_PATH = os.path.join(TRAIN_DIR, "metadata.csv")
LABEL_PATH = os.path.join(TRAIN_DIR, "labels_diff_norm_multi.csv")

# === Read metadata ===
meta_df = pd.read_csv(META_PATH)

# === Determine label from inclination ===
def extract_inclination(sim_string):
    try:
        inc_str = sim_string.split("_")[2]  # e.g. inc0.863
        return float(inc_str.replace("inc", ""))
    except Exception as e:
        print(f"Could not parse inclination from {sim_string}: {e}")
        return None

label_records = []

for _, row in meta_df.iterrows():
    sim_name = row["simulation"]
    inc = extract_inclination(sim_name)
    if inc is None:
        continue

    if 0 <= inc < 3:
        label = 0
    elif 3 <= inc < 9:
        label = 1
    elif 9 <= inc < 18:
        label = 2
    elif 18 <= inc <= 25:
        label = 3
    else:
        raise ValueError(f"Inclination {inc} outside expected range")

    label_records.append({
        "filename": row["diff_path"],
        "simulation": sim_name,
        "co_index": row["co_index"],
        "label": label
    })

# === Save label info ===
pd.DataFrame(label_records).to_csv(LABEL_PATH, index=False)
print(f"Labels saved to {LABEL_PATH}")
