import os
import subprocess
from pathlib import Path

# === CONFIGURATION ===
sim_folder = "sim44_mp2.14_inc0.863_a79.2_phi30.39"
job_script = "mcfost.sbatch"
phantommoddump_exe = "phantommoddump"
phantom_src_dir = Path.home() / "phantom/src/utils"
modfile = phantom_src_dir / "moddump_removeapr.f90"
disc_ids = range(15, 26)  # disc_00015 to disc_00025

# === SET SYSTEM ENVIRONMENT FOR PHANTOM BUILD ===
os.environ["SYSTEM"] = "ifort"

# === ENTER SIMULATION DIRECTORY ===
sim_path = Path(sim_folder).resolve()
os.chdir(sim_path)
print(f"[INFO] Entered directory: {os.getcwd()}")

# === COMPILE PHANTOMMODDUMP WITH INTEL MODULE ===
print("[INFO] Compiling phantommoddump with Intel Fortran (intel-compilers/2022.1.0)...")
try:
    compile_cmd = (
        "module purge && "
        "module load intel-compilers/2022.1.0 && "
        f"make moddump MODFILE={modfile} APR=yes"
    )
    subprocess.run(compile_cmd, shell=True, check=True, executable="/bin/bash")
except subprocess.CalledProcessError as e:
    print(f"[ERROR] Compilation command failed:\n{e}")
    exit(1)

# === VERIFY EXECUTABLE EXISTS ===
exe_path = sim_path / phantommoddump_exe
if not exe_path.exists():
    print(f"[ERROR] Compilation failed: {phantommoddump_exe} was not created.")
    exit(1)
print("[INFO] Compilation successful.")

# === RETURN TO SCRIPT DIRECTORY TO SUBMIT JOB ===
script_dir = Path(__file__).parent.resolve()
os.chdir(script_dir)
print(f"[INFO] Returning to script directory: {script_dir}")

# === SUBMIT JOB WITH --chdir TO RUN IN SIMULATION FOLDER ===
print(f"[INFO] Submitting job script: {job_script} (executing inside {sim_folder})")
try:
    result = subprocess.run(
        f"sbatch --chdir={sim_folder} {job_script}",
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
        universal_newlines=True
    )
    print(f"[INFO] Job submission output: {result.stdout.strip()}")
except subprocess.CalledProcessError as e:
    print(f"[ERROR] Job submission failed: {e}")
    exit(1)

# === REMINDER TO CHECK OUTPUT LATER ===
print("[INFO] Job submitted. Monitor job with `squeue -u $USER` or `sacct`.")
print("[INFO] When job is done, verify output folders:")
for tilt in disc_ids:
    print(f" - dataCO_{tilt}")
    print(f" - dataCO_{tilt}kep")
