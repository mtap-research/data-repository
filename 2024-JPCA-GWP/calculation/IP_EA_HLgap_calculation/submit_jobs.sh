#!/bin/bash

echo type:
read type_index

cd mod/type${type_index}

for index_fn in {1..60}
do
	cd type${type_index}_${index_fn}
	echo type${type_index}_${index_fn}
	sbatch type${type_index}_${index_fn}.slurm
	cd ..
done

cd ../..
