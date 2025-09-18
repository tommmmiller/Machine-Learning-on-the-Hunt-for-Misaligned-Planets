import os
import shutil
import subprocess

# Define absolute base path
BASE_DIR = os.path.abspath(".")
SIMS_DIR = BASE_DIR
PARAMS_FILE = os.path.join(BASE_DIR, "parameters_seed2107632_m5.txt")
SBATCH_FILE = os.path.join(BASE_DIR, "mcfost.sbatch")

# List of simulations to run
sim_files = ['09', '12', '17', '19', '23', '26', '27', '28', '29',
             '36', '40', '41', '43', '44', '45', '48', '49', '50']

# Read parameter lines (skip header)
with open(PARAMS_FILE, "r") as file:
    lines = file.readlines()[1:51]  # Use only lines 1–50 (0-indexed)

# Iterate over parameter sets
for i, line in enumerate(lines):
    params = list(map(float, line.split()))
    if len(params) < 7:
        continue  # Skip malformed lines

    mp, inc, a, view, azi, dist, phi = params
    a *= 3.0
    phi = phi / 360 + 15
    phi_silly = phi + 15

    sim_num = f"{i+1:02d}"
    sim_name = f"sim{sim_num}_mp{mp}_inc{inc}_a{a:.1f}_phi{phi_silly:.2f}"
    sim_dir = os.path.join(SIMS_DIR, sim_name)

    # Skip if simulation is not in selected list
    if sim_num not in sim_files:
        print(f"Skipping {sim_name} (not in simulation list).")
        continue

    if not os.path.isdir(sim_dir):
        print(f"Skipping {sim_name} (directory not found).")
        continue

    try:
        os.chdir(sim_dir)

        # Copy mcfost.sbatch from base directory into simulation folder
        shutil.copy(SBATCH_FILE, "mcfost.sbatch")

        # Submit job using sbatch
        subprocess.run("sbatch mcfost.sbatch", shell=True, check=True)
        print(f"MCFOST {sim_name} started successfully.")

    except subprocess.CalledProcessError as e:
        print(f"Failed to submit job for {sim_name}: {e}")
    except Exception as e:
        print(f"Error processing {sim_name}: {e}")

    finally:
        os.chdir(BASE_DIR)
