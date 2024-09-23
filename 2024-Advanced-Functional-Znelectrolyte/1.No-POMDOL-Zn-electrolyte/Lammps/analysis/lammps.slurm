#!/bin/bash
##SBATCH --partition=amd-gpu
#SBATCH --partition=intel-gold
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --job-name=20ns-no-pomdol

#module purge
module load lammps-17Apr2024

mpirun -n 64 lmp_mpi -i rerun-msd
