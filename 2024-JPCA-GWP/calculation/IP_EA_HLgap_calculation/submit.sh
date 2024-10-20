#!/bin/bash

#SBATCH --partition=amd
##SBATCH -w SG01
##SBATCH --exclude=SG11
#SBATCH --nodes=1
#SBATCH --ntasks=10
##SBATCH --ntasks-per-node=1
##SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=1
##SBATCH --time=10:00:00
#SBATCH --job-name=

export PGI_FASTMATH_CPU=sandybridge
export OMP_THREAD_LIMIT=10

module add gaussian/avx2

cur_dir=$(pwd)

for i in {0..103} 
do
  cd ${cur_dir}
  rm -r mol_${struc}.com
  g16 mol_$i.com
  echo $i "... done!"
done

exit 0

