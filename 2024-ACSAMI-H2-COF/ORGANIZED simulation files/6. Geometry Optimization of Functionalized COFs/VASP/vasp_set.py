import os

if not os.path.exists('calculation'): os.mkdir('calculation')


vasp_dir = os.environ.get('vasp_DIR')
vasp_pbe_dir = os.environ.get('vasp_PBE_DIR')

if vasp_dir:
    print(f'The VASP directory is: {vasp_dir}')
else:
    print('Environment variable vasp_DIR is not set.')

if vasp_pbe_dir:
    print(f'The VASP PBE directory is: {vasp_pbe_dir}')
else:
    print('Environment variable vasp_PBE_DIR is not set.')

# os.environ['OMP_NUM_THREADS'] = 4
os.environ['ASE_VASP_COMMAND'] = f"mpirun -np 1 {vasp_dir}/vasp_std > stdout"
# os.environ['ASE_VASP_COMMAND'] = f"{vasp_dir}/vasp_std > stdout"

os.environ['VASP_PP_PATH'] = vasp_pbe_dir


from ase.io import read, write, Trajectory
from ase.calculators.vasp import Vasp
from ase.optimize import BFGS
# from ase.constraints import FixAtoms

ori_struc_file = 'struc_name'

atoms = read(f'{ori_struc_file}.cif') 

calc = Vasp(
    directory='calculation',  
    xc='PBE', 
    encut=400,                      # 能量切割
    kpts=[1, 1, 1],                 # k点网格
    ibrion=2,                       # 优化设置
    nsw=100,                        # 最大优化步数
    ismear=0,                       # ISMEAR 参数
    sigma=0.1,                      # 费米-狄拉克展宽
    ediff=1e-5,                     # 电子步收敛准则
    ediffg=-0.02,                   # 力收敛准则
    lreal='Auto',                   # REAL空间投影
    lwave=False,                    # 不写WAVECAR
    lcharg=False,                    # 不写CHGCAR
	ivdw=12
)

atoms.calc = calc
calc.write_input(atoms)

trajectory = Trajectory('optimization.traj', 'w', atoms)
opt = BFGS(atoms, trajectory=trajectory)
opt.run(fmax=0.02)  

write(f'{ori_struc_file}_opt.cif', atoms)



