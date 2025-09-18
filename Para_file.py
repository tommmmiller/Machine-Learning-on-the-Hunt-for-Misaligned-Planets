import os 
import shutil
import subprocess

# Define directories and input file
BASE_DIR = "."
SIMS_DIR = os.path.join(BASE_DIR)  # Ensures it goes inside HMP_II
PARAMS_FILE = os.path.join(BASE_DIR, "parameters_seed2107632.txt")
PARA_FILE = os.path.join(BASE_DIR, "ref4.1.para")


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
    phi = phi/360 + 30

    # Format parameters
    sim_num = f"{i+1:02d}"  # Format as '01', '02', etc.
    sim_name = f"sim{sim_num}_mp{mp}_inc{inc}_a{a:.1f}_phi{phi:.2f}"

    # Create simulation directory
    sim_dir = os.path.join(SIMS_DIR, sim_name)
    os.makedirs(sim_dir, exist_ok=True)



    with open(PARA_FILE, "w") as f:
        for line in lines:
            # Replace parameters in para file
            if line.startswith("  0.  45.  3  F           RT: imin, imax, n_incl, centered ?"):
                f.write(f"  {view}.  {view}.  1  F           RT: imin, imax, n_incl, centered ?\n")
            if line.startswith("  0    0.   1             RT: az_min, az_max, n_az angles"):
                f.write(f"  {azi}    {azi}.   1             RT: az_min, az_max, n_az angles\n")
            elif line.startswith("  140.0                   distance (pc)"):
                f.write(f"  {dist}.0                   distance (pc)\n")
            
            

    print(f"Simulation {sim_name} File altered correctly.")
    
    # Change back to the base directory before processing the next simulation
    os.chdir("..")
    

