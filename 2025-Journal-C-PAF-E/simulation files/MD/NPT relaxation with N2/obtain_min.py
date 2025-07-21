import os
import numpy as np
from ase.io import read, write
from tqdm import tqdm
import sys

list_file = f'list_{sys.argv[1]}'
output_file_dir = 'self_output'
traj_dir = 'traj'
opt_output_dir = 'npt_output'
# element_mapping = {'C':'Cl','H': 'C', 'He': 'H', 'Li': 'C', 'Be': 'C', 'B': 'N'}
element_mapping = {'H': 'C', 'He': 'H', 'Li': 'C', 'Be': 'C'}

with open(f'{list_file}', 'r', encoding='utf-8') as file:
    lines = file.readlines()
struc_list = [f'{line.strip()}_77' for line in lines]
# struc_list = ['wkf_2_opt_77','vca_1_opt_77','wmy_0_opt_77']
# struc_list = os.listdir(output_file_dir)
print(struc_list)
for struc_name in tqdm(struc_list):
    print(struc_name)
    for traj_file in os.listdir(os.path.join(output_file_dir,struc_name)):
        traj_name = traj_file.split('.')[0]
        output_file = os.path.join(output_file_dir,struc_name,f'{traj_name}.txt')
        trajectory_file = os.path.join(traj_dir,struc_name,f'{traj_name}.lammpstrj')

        with open(output_file, "r") as file:
            lines = file.readlines()

        vol_change = np.array([float(line.split()[11]) for line in lines[1:]])
        hist, bin_edges = np.histogram(vol_change, bins=200, density=True)
        max_bin_index = np.argmax(hist)
        mode_value = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2
        frame_number = (np.abs(vol_change - mode_value)).argmin()

        atoms = read(trajectory_file, index=frame_number)
        atoms = atoms[[atom.symbol in ['H', 'He', 'Li', 'Be'] for atom in atoms]]
        # atoms = atoms[[atom.symbol in ['C','H', 'He', 'Li', 'Be','B'] for atom in atoms]]

        new_symbols = [element_mapping.get(symbol, symbol) for symbol in atoms.get_chemical_symbols()]
        atoms.set_chemical_symbols(new_symbols)
        write(os.path.join(opt_output_dir,f'{traj_name}_opt.cif'), atoms)

