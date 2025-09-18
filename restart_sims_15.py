import os
import shutil
import subprocess
import re

# Define directories and input file
BASE_DIR = "."
SIMS_DIR = os.path.join(BASE_DIR)
PARAMS_FILE = os.path.join(BASE_DIR, "parameters_seed2107632.txt")
FILES_TO_COPY = ["run.sbatch", "disc.in", "disc.setup"]

# Ensure base simulations directory exists
os.makedirs(SIMS_DIR, exist_ok=True)

# Read the file and process lines (skip first line)
with open(PARAMS_FILE, "r") as file:
    lines = file.readlines()[1:51]

# Loop through parameter sets
for i, line in enumerate(lines):
    params = list(map(float, line.split()))
    if len(params) < 7:
        continue

    mp, inc, a, view, azi, dist, phi = params
    a *= 3.0
    sim_num = f"{i+1:02d}"
    sim_name = f"sim{sim_num}_mp{mp}_inc{inc}_a{a:.1f}_phi{phi}"
    sim_dir = os.path.join(SIMS_DIR, sim_name)

    # Make simulation directory if it doesn't exist
    os.makedirs(sim_dir, exist_ok=True)

    # Copy input files
    for file_name in FILES_TO_COPY:
        src_path = os.path.join(BASE_DIR, file_name)
        dst_path = os.path.join(sim_dir, file_name)
        shutil.copy(src_path, dst_path)

    # Modify disc.in
    disc_in_path = os.path.join(sim_dir, "disc.in")
    with open(disc_in_path, "r") as f:
        lines = f.readlines()

    with open(disc_in_path, "w") as f:
        for line in lines:
            if "dumpfile =" in line:
                f.write("            dumpfile =  disc_00015    ! dump file to start from\n")
            elif "nmaxdumps =" in line:
                f.write("           nmaxdumps =          10    ! stop after n full dumps (-ve=ignore)\n")
            
            elif "dtmax" in line:
                match = re.search(r"dtmax\s*=\s*([0-9.eE+-]+)", line)
                if match:
                    old_dtmax = float(match.group(1))
                    new_dtmax = old_dtmax / 10
                    comment = line.split("!")[-1].strip() if "!" in line else ""
                    f.write(f"               dtmax =  {new_dtmax:.5f}    ! {comment}\n")
                else:
                    f.write(line)
          

            else:
                f.write(line)

    # Modify run.sbatch
    sbatch_file = os.path.join(sim_dir, "run.sbatch")
    with open(sbatch_file, "r") as f:
        lines = f.readlines()

    with open(sbatch_file, "w") as f:
        for line in lines:
            if line.startswith("#SBATCH --job-name="):
                f.write(f"#SBATCH --job-name={sim_name}\n")
            else:
                f.write(line)
    """
    # Submit the job
    os.chdir(sim_dir)
    subprocess.run("sbatch run.sbatch", shell=True, check=True)
    os.chdir(BASE_DIR)
    """
    print(f"Simulation {sim_name} restarted successfully.")
