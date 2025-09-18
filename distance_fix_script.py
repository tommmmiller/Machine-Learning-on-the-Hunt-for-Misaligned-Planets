import os

# Configuration
BASE_DIR = os.path.abspath(".")
PARAMS_FILE = os.path.join(BASE_DIR, "parameters_seed2107632_m5.txt")
SKIP_FILE = os.path.join(BASE_DIR, "skipped_simulations.txt")

# Load skipped simulation names
with open(SKIP_FILE, "r") as f:
    skipped_lines = f.readlines()[1:]
    skipped_sims = set(line.split()[0] for line in skipped_lines)

# Load parameter lines
with open(PARAMS_FILE, "r") as f:
    lines = f.readlines()[1:51]

for i, line in enumerate(lines):
    parts = list(map(float, line.strip().split()))
    if len(parts) < 7:
        continue

    mp, inc, a, view, azi, dist, phi = parts
    a *= 3.0
    phi_silly = phi / 360 + 30  # keep naming consistent
    sim_num = f"{i+1:02d}"
    sim_name = f"sim{sim_num}_mp{mp}_inc{inc}_a{a:.1f}_phi{phi_silly:.2f}"
    sim_dir = os.path.join(BASE_DIR, sim_name)

    if sim_name in skipped_sims:
        print(f"Skipping {sim_name} (in skipped list)")
        continue
    if not os.path.isdir(sim_dir):
        print(f"Skipping {sim_name} (missing folder)")
        continue

    para_path = os.path.join(sim_dir, "ref4.1.para")
    if not os.path.isfile(para_path):
        print(f"Skipping {sim_name} (missing ref4.1.para)")
        continue

    # Read and replace only the distance line
    with open(para_path, "r") as f:
        lines_out = []
        for L in f:
            if "distance (pc)" in L:
                lines_out.append(f"{dist:.1f}                   distance (pc)\n")
            else:
                lines_out.append(L)

    with open(para_path, "w") as f:
        f.writelines(lines_out)

    print(f"Patched distance in {sim_name}")
