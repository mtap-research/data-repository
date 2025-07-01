import os
import numpy as np
from ase.io import read, write
from tqdm import tqdm
import sys

list_file = f'lists/list_{sys.argv[1]}'
output_file_dir = 'npt_output'
traj_dir = 'traj'
opt_output_dir = 'opt_npt'
element_mapping = {'H': 'C', 'Li': 'C', 'Be': 'C', 'He': 'H'}

with open(f'{list_file}', 'r', encoding='utf-8') as file:
    lines = file.readlines()
struc_list = [line.strip() for line in lines]

for struc_name in tqdm(struc_list):
    if not os.path.exists(os.path.join(opt_output_dir, f'{struc_name}_opt.cif')):
        output_file = os.path.join(output_file_dir,f'{struc_name}_output.txt')
        trajectory_file = os.path.join(traj_dir,f'{struc_name}.lammpstrj')

        with open(output_file, "r") as file:
            lines = file.readlines()

        vol_change = np.array([float(line.split()[6]) for line in lines[1:]])
        hist, bin_edges = np.histogram(vol_change, bins=2000, density=True)
        max_bin_index = np.argmax(hist)
        mode_value = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2
        frame_number = (np.abs(vol_change - mode_value)).argmin()

        atoms = read(trajectory_file, index=frame_number)
        new_symbols = [element_mapping.get(symbol, symbol) for symbol in atoms.get_chemical_symbols()]
        atoms.set_chemical_symbols(new_symbols)
        write(os.path.join(opt_output_dir,f'{struc_name}_opt.cif'), atoms)
