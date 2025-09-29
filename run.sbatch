#!/bin/bash
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --job-name=test4
#SBATCH --output=disc.in.qout
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tom.miller@warwick.ac.uk
#SBATCH --time=048:00:00
#SBATCH --mem-per-cpu=3700
echo "HOSTNAME = $HOSTNAME"
echo "HOSTTYPE = $HOSTTYPE"
echo Time is `date`
echo Directory is `pwd`

module purge
module load iccifort/2020.4.304
export SYSTEM=ifort

ulimit -s unlimited
export OMP_SCHEDULE="dynamic"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_STACKSIZE=1024m

echo "starting phantom run..."
export outfile=`grep logfile "disc.in" | sed "s/logfile =//g" | sed "s/\\!.*//g" | sed "s/\s//g"`
echo "writing output to $outfile"
./phantom disc.in >& $outfile
