
from tqdm import tqdm

import torch
import os
from chgnet.model import StructOptimizer
from pymatgen.io.cif import CifWriter
from pymatgen.core import Structure

torch.has_cuda=torch.backends.cuda.is_built()
torch.has_cudnn=torch.backends.cudnn.is_available()
torch.has_mps=torch.backends.mps.is_built()
torch.has_mkldnn=torch.backends.mkldnn.is_available()
with open("list_single.txt", "r") as f:
    mofs = f.readlines()
relaxer = StructOptimizer(optimizer_class="LBFGS")
fail = []
for mof in tqdm(mofs):
    try:
        mof = mof.replace("\n","")
        structure = Structure.from_file(os.path.join("../mod/single",f"{mof}.cif"))
        result = relaxer.relax(structure,fmax=0.02,steps=50000,relax_cell=True,loginterval=1)
        cif_writer = CifWriter(result["final_structure"])
        cif_writer.write_file(os.path.join("opt",f"{mof}_chgnet.cif"))
    except:
        print(mof,"fail")
#         fail.append(mof)
# np.savetxt("fail.txt",fail,format=)
