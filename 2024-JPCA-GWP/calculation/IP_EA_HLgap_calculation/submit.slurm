#!/bin/bash

##SBATCH --partition=amd
##SBATCH -w SG01
##SBATCH --exclude=SG11
#SBATCH --nodes=1
#SBATCH --ntasks=10
##SBATCH --ntasks-per-node=1
##SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=1
##SBATCH --time=10:00:00
#SBATCH --job-name=gwp_neutral

export PGI_FASTMATH_CPU=sandybridge
export OMP_THREAD_LIMIT=10

module add gaussian/avx2
rm *.log *.chk

calc_type=neutral

while read name
do
	cd ./input_files/${calc_type}/${name}_${calc_type}
	g16 ${name}_${calc_type}.gjf &
	cd ../..
done<list

sleep 4444444444



exit 0