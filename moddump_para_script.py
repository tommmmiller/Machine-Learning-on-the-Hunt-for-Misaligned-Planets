import os
import shutil
import subprocess

# === Configuration ===
BASE_DIR = os.path.abspath(".")
PARAMS_FILE = os.path.join(BASE_DIR, "parameters_seed2107632_m5.txt")
REFPARA_TEMPLATE = os.path.join(BASE_DIR, "ref4.1.para")
SKIP_FILE = os.path.join(BASE_DIR, "skipped_simulations.txt")

# === Lines to Match and Replace Exactly ===
INCL_LINE_MATCH = "30.  30.  1  F           RT: imin, imax, n_incl, centered ?"
AZIM_LINE_MATCH = "0    0.   1             RT: az_min, az_max, n_az angles"
DIST_LINE_MATCH = "140.0                   distance (pc)"

# --- Load skipped simulation names ---
with open(SKIP_FILE, "r") as f:
    skipped_lines = f.readlines()[1:]
    skipped_sims = set(line.split()[0] for line in skipped_lines)

# --- Load parameter lines ---
with open(PARAMS_FILE, "r") as f:
    lines = f.readlines()[1:51]

for i, line in enumerate(lines):
    parts = list(map(float, line.strip().split()))
    if len(parts) < 7:
        continue

    mp, inc, a, view, azi, dist, phi = parts
    a *= 3.0
    phi = phi/360 + 30
    sim_num = f"{i+1:02d}"
    sim_name = f"sim{sim_num}_mp{mp}_inc{inc}_a{a:.1f}_phi{phi}"
    sim_dir = os.path.join(BASE_DIR, sim_name)

    if sim_name in skipped_sims:
        print(f"Skipping {sim_name} (in skipped list)")
        continue

    if not os.path.isdir(sim_dir):
        print(f"Skipping {sim_name} (folder missing)")
        continue

    # === Run make moddump with DEBUG flag ===
    print(f"Running moddump with DEBUG in {sim_name} ...")
    try:
        subprocess.run(
            ["make", "moddump", "MODFILE=moddump_removeapr.f90", "APR=yes", "DEBUG=yes"],
            check=True,
            cwd=sim_dir
        )
        print(f"moddump built in {sim_name} with DEBUG")
    except subprocess.CalledProcessError as e:
        print(f"moddump failed in {sim_name}: {e}")
        continue

    # === Copy and patch ref4.1.para ===
    target_para = os.path.join(sim_dir, "ref4.1.para")
    shutil.copy(REFPARA_TEMPLATE, target_para)

    with open(target_para, "r") as f:
        lines_out = []
        for L in f:
            if L.strip() == INCL_LINE_MATCH:
                lines_out.append(f"{view:.1f}  {view:.1f}  1  F           RT: imin, imax, n_incl, centered ?\n")
            elif L.strip() == AZIM_LINE_MATCH:
                lines_out.append(f"{azi:.1f}  {azi:.1f}  1             RT: az_min, az_max, n_az angles\n")
            elif L.strip() == DIST_LINE_MATCH:
                lines_out.append(f"{dist:.1f}                   distance (pc)\n")
            else:
                lines_out.append(L)

    with open(target_para, "w") as f:
        f.writelines(lines_out)

    print(f"Patched ref4.1.para in {sim_name}")
