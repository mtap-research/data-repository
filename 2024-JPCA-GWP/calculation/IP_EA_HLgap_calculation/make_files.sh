cur_dir=$(pwd)

while read struc
do
	cd ${cur_dir}/input_files/neutral
	rm -r mol_${struc}_neutral
      	mkdir mol_${struc}_neutral
      	cd mol_${struc}_neutral
	echo "%chk=neutral_${struc}.chk" > mol_${struc}_neutral.gjf
      	echo "%nprocs=4" >> mol_${struc}_neutral.gjf
      	echo "# opt=calcfc b3lyp/6-31g(d,p)" >> mol_${struc}_neutral.gjf
      	echo "" >> mol_${struc}_neutral.gjf
      	echo "mol_${struc}" >> mol_${struc}_neutral.gjf
      	echo "" >> mol_${struc}_neutral.gjf
      	echo "0 1" >> mol_${struc}_neutral.gjf
      	cat ${cur_dir}/xyz_files/${struc}.xyz | tail -n+3  >> mol_${struc}_neutral.gjf
      	echo "" >> mol_${struc}_neutral.gjf
      	echo "" >> mol_${struc}_neutral.gjf

	cd ${cur_dir}/input_files/anion
	rm -r mol_${struc}_anion
        mkdir mol_${struc}_anion
        cd mol_${struc}_anion
	echo "%chk=anion_${struc}.chk" > mol_${struc}_anion.gjf
        echo "%nprocs=4" >> mol_${struc}_anion.gjf
        echo "# opt=calcfc b3lyp/6-31g(d,p)" >> mol_${struc}_anion.gjf
        echo "" >> mol_${struc}_anion.gjf
        echo "mol_${struc}" >> mol_${struc}_anion.gjf
        echo "" >> mol_${struc}_anion.gjf
        echo "-1 2" >> mol_${struc}_anion.gjf
        cat ${cur_dir}/xyz_files/${struc}.xyz | tail -n+3  >> mol_${struc}_anion.gjf
        echo "" >> mol_${struc}_anion.gjf
        echo "" >> mol_${struc}_anion.gjf
	
	cd ${cur_dir}/input_files/cation
	rm -r mol_${struc}_cation
      	mkdir mol_${struc}_cation
        cd mol_${struc}_cation
  echo "%chk=cation_${struc}.chk" > mol_${struc}_cation.gjf
        echo "%nprocs=4" >> mol_${struc}_cation.gjf
        echo "# opt=calcfc b3lyp/6-31g(d,p)" >> mol_${struc}_cation.gjf
        echo "" >> mol_${struc}_cation.gjf
        echo "mol_${struc}" >> mol_${struc}_cation.gjf
        echo "" >> mol_${struc}_cation.gjf
        echo "1 2" >> mol_${struc}_cation.gjf
        cat ${cur_dir}/xyz_files/${struc}.xyz | tail -n+3  >> mol_${struc}_cation.gjf
        echo "" >> mol_${struc}_cation.gjf
        echo "" >> mol_${struc}_cation.gjf

	cd ${cur_dir}

done < ${cur_dir}/list_all.txt
