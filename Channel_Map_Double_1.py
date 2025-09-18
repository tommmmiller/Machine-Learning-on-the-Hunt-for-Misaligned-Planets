import os
import subprocess
from shutil import copy2

# Define sim folder and working filenames
sim_dir = "sim01_mp1.41_inc21.5_a188.4_phi30.66"
base_dir = os.getcwd()
full_sim_path = os.path.join(base_dir, sim_dir)
os.chdir(full_sim_path)

phantom_root  = os.path.expanduser("~/phantom")
phantom_build = os.path.join(phantom_root, "build")

# Copy moddump_removeapr.f90 from central Phantom utils directory
moddump_src   = os.path.join(phantom_root, "src/utils/moddump_removeapr.f90")
moddump_target = os.path.join(full_sim_path, "moddump_removeapr.f90")
copy2(moddump_src, moddump_target)
with open("Makefile", "w") as f:
    f.write("SETUP    = disc\n")
    f.write(f"RUNDIR   = {full_sim_path}\n")
    f.write(f"BUILDDIR = {phantom_build}\n")
    f.write(f"include  {os.path.join(phantom_root, 'Makefile')}\n")


# Compile phantommoddump with APR=yes
subprocess.run("make moddump MODFILE=moddump_removeapr.f90 APR=yes", shell=True, check=True)

# Convert file disc_00014 to non-APR file
input_file = "disc_00014"
output_file = "CM_sim01_14"
subprocess.run(f"./phantommoddump {input_file} {output_file}", shell=True, check=True)

# Copy parameter file from another sim dir
para_src = "../sim10_mp3.84_inc8.1_a66.6_phi30.23/ref4.1.para"
para_target = os.path.join(full_sim_path, "ref4.1.para")
copy2(para_src, para_target)

# Prepare two MCFOST commands: with and without Keplerian velocity
cmds = [
    f"mcfost ref4.1.para -phantom {output_file}_00000 -mol -turn-off_planets",
    f"mcfost ref4.1.para -phantom {output_file}_00000 -mol -turn-off_planets -vphi_Kep"
]

# Save to two shell scripts for Slurm jobs
for i, cmd in enumerate(cmds):
    job_script = f"mcfost_job_{i+1}.sh"
    with open(job_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --nodes=1\n")
        f.write("#SBATCH --ntasks-per-node=1\n")
        f.write("#SBATCH --cpus-per-task=24\n")
        f.write(f"#SBATCH --job-name=mcfost_{i+1}\n")
        f.write(f"#SBATCH --output=mcfost_{i+1}.out\n")
        f.write("#SBATCH --time=24:00:00\n")
        f.write("#SBATCH --mem=80G\n")
        f.write("#SBATCH --mail-type=END,FAIL\n")
        f.write("#SBATCH --mail-user=tom.miller@warwick.ac.uk\n")
        f.write("\n")
        f.write("module purge\n")
        f.write("source ~/mcfost_utils/load_mcfost.sh\n")
        f.write("ulimit -s unlimited\n")
        f.write("export OMP_SCHEDULE='dynamic'\n")
        f.write("export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n")
        f.write("export OMP_STACKSIZE=1024m\n")
        f.write("\n")
        f.write(f"echo Running command: {cmd}\n")
        f.write(f"{cmd}\n")