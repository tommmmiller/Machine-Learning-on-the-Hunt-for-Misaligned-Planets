import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend suitable for clusters
import matplotlib.pyplot as plt
import mcfost

plt.ioff()  # Turn off interactive mode

# -- Simulation Parameters --
PARAMS_FILE = "parameters_seed2107632_m5.txt"
SCRIPTS_BASE = os.getcwd()

# Simulations to run
sim_files = ['09', '12', '17', '19', '23', '26', '27', '28', '29',
             '36', '40', '41', '43', '44', '45', '48', '49', '50']

# Load parameter lines
with open(PARAMS_FILE, "r") as file:
    lines = file.readlines()[1:51]  # Skip header

# Loop over simulations
for i, line in enumerate(lines):
    sim_num = f"{i+1:02d}"
    if sim_num not in sim_files:
        continue

    try:
        mp, inc, a, view, azi, dist, phi = map(float, line.split())
    except ValueError:
        print(f"[Line {i+1}] Malformed, skipping.")
        continue

    a *= 3.0
    phi_silly = phi / 360 + 15 + 15
    sim_name = f"sim{sim_num}_mp{mp}_inc{inc}_a{a:.1f}_phi{phi_silly:.2f}"
    sim_dir = os.path.join(SCRIPTS_BASE, sim_name)

    if not os.path.isdir(sim_dir):
        print(f"[{sim_name}] Directory missing, skipping.")
        continue

    print(f"Processing {sim_name}")
    os.chdir(sim_dir)

    for j in range(15, 26):
        folder1 = f"data_CO_{j}"
        folder2 = f"data_CO_{j}kep"
        if not (os.path.isdir(folder1) and os.path.isdir(folder2)):
            continue

        try:
            mol = mcfost.Line(folder1)
            mol_kep = mcfost.Line(folder2)
        except Exception as e:
            print(f" - Error loading CO_{j}: {e}")
            continue

        v_range = np.linspace(-3, 3, 20)
        fmin, fmax = 4, 66
        bmaj = bmin = 0.01
        bpa = -88

        maximums = []
        for v in v_range:
            try:
                mol.plot_map(0, v=v, Tb=True, cmap='inferno', fmin=fmin, fmax=fmax,
                             bmaj=bmaj, bmin=bmin, bpa=bpa, plot_beam=True)
                plt.close()
                mol_kep.plot_map(0, v=v, Tb=True, cmap='inferno', fmin=fmin, fmax=fmax,
                                 bmaj=bmaj, bmin=bmin, bpa=bpa, plot_beam=True)
                plt.close()
                diff = mol_kep.last_image - mol.last_image
                maximums.append(np.max(np.abs(diff)))
            except Exception as e:
                print(f" - Plot error v={v:.2f}: {e}")
                maximums.append(0)

        if not maximums:
            continue

        max_vel = v_range[np.argmax(maximums)]
        try:
            mol.plot_map(0, v=max_vel, Tb=True, cmap='inferno', fmin=fmin, fmax=fmax,
                         bmaj=bmaj, bmin=bmin, bpa=bpa, plot_beam=True)
            mol_kep.plot_map(0, v=max_vel, Tb=True, cmap='inferno', fmin=fmin, fmax=fmax,
                             bmaj=bmaj, bmin=bmin, bpa=bpa, plot_beam=True)
            diff = mol_kep.last_image - mol.last_image
            max_abs = np.percentile(np.abs(diff), 99)

            plt.figure(figsize=(6, 5))
            im = plt.imshow(diff, origin='lower', cmap='RdBu_r',
                            vmin=-max_abs, vmax=+max_abs)
            plt.colorbar(im, label='Difference (K)')
            plt.title(f"{sim_name} ΔT_b CO_{j}\n(v = {max_vel:.2f} km/s)")
            plt.xlabel("Pixel X")
            plt.ylabel("Pixel Y")
            outname = f"diff_map_CO_{j}.png"
            plt.savefig(outname, dpi=150, bbox_inches='tight')
            plt.close()
            print(f" - Saved {outname}")
        except Exception as e:
            print(f" - Final plot error CO_{j}: {e}")
            plt.close()

    os.chdir(SCRIPTS_BASE)
