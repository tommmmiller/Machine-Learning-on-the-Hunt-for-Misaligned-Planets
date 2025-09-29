
import os 
import shutil
import subprocess

# Define directories and input file
BASE_DIR = "."
SIMS_DIR = os.path.join(BASE_DIR)  # Ensures it goes inside HMP_II
PARAMS_FILE = os.path.join(BASE_DIR, "parameters_seed2107632_m5.txt")
FILES_TO_COPY = [os.path.join(BASE_DIR, "run.sbatch"),
                 os.path.join(BASE_DIR, "disc.in"),
                 os.path.join(BASE_DIR, "disc.setup")]

# Ensure all_sims directory exists
os.makedirs(SIMS_DIR, exist_ok=True)


# Read the file and process each line, skipping the first
with open(PARAMS_FILE, "r") as file:
    lines = file.readlines()[1:51]  # Skip first line

# Iterate over parameter sets
for i, line in enumerate(lines):
    # Parse parameters
    params = list(map(float, line.split()))
    if len(params) < 7:
        continue  # Skip invalid lines


    mp, inc, a, view, azi, dist, phi = params
    a=a*3.0
    phi = phi/360 + 15
    phi_silly = phi+15
 # Format parameters
    sim_num = f"{i+1:02d}"  # Format as '01', '02', etc.
    sim_name = f"sim{sim_num}_mp{mp}_inc{inc}_a{a:.1f}_phi{phi_silly:.2f}"
    sim_dir = os.path.join(SIMS_DIR, sim_name)

    last_file = os.path.join(sim_dir, "disc_00015")
    if os.path.exists(last_file):
        print(f"Simulation {sim_name} is already complete. Skipping...")
        continue  # Skip if the last file exists

    # Ensure we are inside the correct directory
    os.chdir(sim_dir)

    disc_in = "disc.in"
    with open(disc_in, "r") as f:
        lines = f.readlines()
    with open(disc_in, "w") as f:
        for line in lines:
            if line.startswith("           nfulldump ="):
                f.write("           nfulldump =          10    ! full dump every n dumps\n")
            else:
                f.write(line)

    # Submit the simulation job
    subprocess.run("sbatch run.sbatch", shell=True, check=True)

    print(f"Simulation {sim_name} restarted successfully.")



    # Change back to the base directory before processing the next simulation
    os.chdir("..")
