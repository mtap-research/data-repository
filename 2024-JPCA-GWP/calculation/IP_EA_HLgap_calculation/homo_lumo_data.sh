rm homo_data
rm lumo_data

for i in {0..199}
do
    grep occ neutral_$i.log | tail -n 1 | awk '{print $NF}' >>homo_data
    grep virt neutral_$i.log | head -n 1 | awk '{print $5}' >>lumo_data
    echo $i
done

paste homo_data lumo_data > h_l_data
