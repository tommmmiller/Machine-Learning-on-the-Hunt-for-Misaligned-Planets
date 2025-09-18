import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymcfost as mcfost
import pandas as pd
import logging

plt.ioff()

# === Paths ===
BASE_DIR = os.getcwd()  # where simXX_mp... folders live
TRAIN_DIR = os.path.join(BASE_DIR, "train_all")
RAW_DIR = os.path.join(TRAIN_DIR, "raw")
DIFF_DIR = os.path.join(TRAIN_DIR, "diff")
META_PATH = os.path.join(TRAIN_DIR, "metadata.csv")
LOG_PATH = os.path.join(TRAIN_DIR, "generation_log.txt")

# === Simulation groups ===
sim_files_A = ['01','04','06','07','10','13','14','15',
               '20','25','30','37','38','39']
sim_files_B = ['09','12','17','19','23','26','27','28','29',
               '36','40','41','43','44','45','48','49','50']

# === Constants ===
VEL_RANGE = np.linspace(-3, 3, 20)
CX, CY, R = 150, 150, 5
Y, X = np.ogrid[:301, :301]
MASK = (X - CX)**2 + (Y - CY)**2 <= R**2

# === Directory setup ===
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(DIFF_DIR, exist_ok=True)

# === Logging ===
logging.basicConfig(
    filename=LOG_PATH,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
log = logging.getLogger()
metadata_records = []

# === Find all simulation folders ===
sim_folders = [d for d in os.listdir(BASE_DIR)
               if d.startswith("sim") and os.path.isdir(os.path.join(BASE_DIR, d))]

# Sort by sim number
sim_folders.sort(key=lambda name: int(re.search(r"sim(\d+)", name).group(1)))

for sim_folder in sim_folders:
    sim_match = re.search(r"sim(\d+)", sim_folder)
    if not sim_match:
        continue
    sim_num = sim_match.group(1).zfill(2)
    sim_path = os.path.join(BASE_DIR, sim_folder)

    # Decide folder template
    if sim_num in sim_files_A:
        folder_template1 = lambda j: os.path.join(sim_path, f"data_CO_{j}/data_CO")
        folder_template2 = lambda j: os.path.join(sim_path, f"data_CO_{j}kep_vr")
    elif sim_num in sim_files_B:
        folder_template1 = lambda j: os.path.join(sim_path, f"data_CO_{j}")
        folder_template2 = lambda j: os.path.join(sim_path, f"data_CO_{j}kep_vr")
    else:
        log.warning(f"{sim_folder}: Not in A or B list, skipping.")
        continue

    log.info(f"Processing {sim_folder}")
    os.chdir(sim_path)

    for j in range(15, 26):
        folder1 = folder_template1(j)
        folder2 = folder_template2(j)

        if not (os.path.isdir(folder1) and os.path.isdir(folder2)):
            log.warning(f"{sim_folder}: Missing {folder1} or {folder2}, skipping.")
            continue

        try:
            mol = mcfost.Line(folder1)
            mol_kep = mcfost.Line(folder2)
        except Exception as e:
            log.error(f"{sim_folder}: Error loading CO_{j}: {e}")
            continue

        for idx, vel in enumerate(VEL_RANGE):
            try:
                mol.plot_map(0, v=vel, Tb=True, cmap="inferno",
                             fmin=4, fmax=66, bmaj=0.01, bmin=0.01,
                             bpa=-88, plot_beam=True)
                plt.close()
                mol_kep.plot_map(0, v=vel, Tb=True, cmap="inferno",
                                 fmin=4, fmax=66, bmaj=0.01, bmin=0.01,
                                 bpa=-88, plot_beam=True)
                plt.close()

                data = mol.last_image.copy()
                data2 = mol_kep.last_image.copy()
                if data.shape != (301,301) or data2.shape != (301,301):
                    continue

                data[MASK] = 0.0
                data2[MASK] = 0.0
                diff = data2 - data

                raw_filename = f"{sim_folder}_raw_{j}_idx{idx}.npy"
                diff_filename = f"{sim_folder}_diff_{j}_idx{idx}.npy"
                np.save(os.path.join(RAW_DIR, raw_filename), data)
                np.save(os.path.join(DIFF_DIR, diff_filename), diff)

                metadata_records.append({
                    "simulation": sim_folder,
                    "co_index": j,
                    "vel_index": idx,
                    "velocity": vel,
                    "raw_path": raw_filename,
                    "diff_path": diff_filename
                })

                log.info(f"{sim_folder}: Saved {raw_filename}, {diff_filename}")
            except Exception as e:
                log.error(f"{sim_folder}: Error at CO_{j}, v={vel:.2f}: {e}")
                plt.close()

# Save metadata
df_meta = pd.DataFrame(metadata_records)
df_meta.to_csv(META_PATH, index=False)
log.info(f"Metadata saved to {META_PATH}")
print("Processing complete. Metadata saved.")
