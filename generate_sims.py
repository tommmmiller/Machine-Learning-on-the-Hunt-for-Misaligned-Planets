import os 
import shutil
import subprocess

# Define directories and input file
BASE_DIR = "/Users/tommiller/Desktop/HMP_II"
SIMS_DIR = os.path.join(BASE_DIR, "all_sims_test")  # Ensures it goes inside HMP_II
PARAMS_FILE = os.path.join(BASE_DIR, "parameters_seed2107632.txt")
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
    # Format parameters
    sim_num = f"{i+1:02d}"  # Format as '01', '02', etc.
    sim_name = f"sim{sim_num}_mp{mp}_inc{inc}_a{a:.1f}_phi{phi}"

    # Create simulation directory
    sim_dir = os.path.join(SIMS_DIR, sim_name)
    os.makedirs(sim_dir, exist_ok=True)
    
    # Copy files into new directory
    for file in FILES_TO_COPY:
        shutil.copy(file, sim_dir)

    # Modify disc.setup with new parameters
    setup_file = os.path.join(sim_dir, "disc.setup")

    # Read and modify the disc.setup file
    with open(setup_file, "r") as f:
        lines = f.readlines()

    with open(setup_file, "w") as f:
        for line in lines:
            # Replace parameters in disc.setup
            if line.startswith("            mplanet1 ="):
                f.write(f"            mplanet1 =       {mp}    ! planet mass (in Jupiter mass)\n")
            elif line.startswith("         inclplanet1 ="):
                f.write(f"         inclplanet1 =       {inc}    ! planet orbital inclination (deg)\n")
            elif line.startswith("            rplanet1 ="):
                f.write(f"            rplanet1 =        {a:.1f}    ! planet distance from star\n")
            elif line.startswith("             posangl ="):
                f.write(f"             posangl =       {phi}    ! position angle (deg)\n")
            else:
                f.write(line) 
            

    sbatch_file = os.path.join(sim_dir, "run.sbatch")

    # Read and modify run.sbatch
    with open(sbatch_file, "r") as f:
        lines = f.readlines()

    with open(sbatch_file, "w") as f:
        for line in lines:
            if line.startswith("#SBATCH --job-name="):
                f.write(f"#SBATCH --job-name={sim_name}\n")  # Replace file_name with directory name
            else:
                f.write(line)  # Keep other lines unchanged
    """

    try:
        print(f"Running setup for {sim_name} in {sim_dir}...")

        # Ensure we are inside the correct directory
        os.chdir(sim_dir)

        # Run Phantom setup commands
        subprocess.run("~/phantom/scripts/writemake.sh disc > Makefile", shell=True, check=True)
        subprocess.run("make APR=yes", shell=True, check=True)
        subprocess.run("make setup APR=yes", shell=True, check=True)
        subprocess.run("./phantomsetup disc", shell=True, check=True)

        # Submit the simulation job
        subprocess.run("sbatch run.sbatch", shell=True, check=True)

        print(f"Simulation {sim_name} started successfully.")
    
    except subprocess.CalledProcessError as e:
        print(f"Error running setup for {sim_name}: {e}")
    
    finally:
        # Change back to the base directory before processing the next simulation
        os.chdir(BASE_DIR)
    """