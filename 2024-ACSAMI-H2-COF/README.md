# Computational Exploration of Adsorption-based Hydrogen Storage in Mg-alkoxide Functionalized Covalent-Organic Frameworks (COFs): Force-field and Machine Learning Models

* Publication: (to-be-updated)
  
* DOI: to-be-updated

This repository includes all the input files and some example output files for GCMC simulation, computational structural models of functionalized COFs, geometry optimizations, and the raw data for plotting the figures in the manuscript (simulation-related sections).

This repository was created by Yu Chen.

## Files Directory:
```
.
├── 1. Functionalized COFs
│   ├── README.txt
│   ├── func_COF_label.zip
│   └── func_COF_unlabel.zip
├── 2. MP2 calculations
│   ├── one H2 with catecholate
│   │   ├── opt
│   │   │   ├── one.gjf
│   │   │   ├── one.log
│   │   │   ├── two.gjf
│   │   │   └── two.log
│   │   └── single-point
│   │       ├── one.gjf
│   │       ├── one.log
│   │       ├── two.gjf
│   │       └── two.log
│   └── two H2 with catecholate
│       ├── opt
│       │   ├── one_3_frag.gjf
│       │   ├── one_3_frag.log
│       │   ├── one_3_frag_2.gjf
│       │   ├── one_3_frag_2.log
│       │   ├── two_3_frag.gjf
│       │   ├── two_3_frag.log
│       │   ├── two_3_frag_2.gjf
│       │   └── two_3_frag_2.log
│       └── single-point
│           ├── one_3_frag.gjf
│           ├── one_3_frag.log
│           ├── one_3_frag_2.gjf
│           ├── one_3_frag_2.log
│           ├── two_3_frag.gjf
│           ├── two_3_frag.log
│           ├── two_3_frag_2.gjf
│           └── two_3_frag_2.log
├── 3. FF Fitting
│   ├── E_LJ_calc.ipynb
│   ├── FF_LJ_coulomb_params.csv
│   ├── FF_fitting-arranged-ver2.ipynb
│   ├── LJ-Coulomb_E_bind.csv
│   ├── MP2_E_bind.xlsx
│   └── README.md
├── 4. H2 - GCMC
│   ├── example output files
│   │   └── output_21320N2_ddec_single_all_1.1.4_111.000000_1e+07.data
│   └── input files
│       ├── H2.def
│       ├── forcefield
│       │   ├── COF_ori
│       │   │   ├── force_field_mixing_rules.def
│       │   │   └── pseudo_atoms.def
│       │   ├── O-Mg-O_morse_newFF_single
│       │   │   ├── force_field.def
│       │   │   ├── force_field_mixing_rules.def
│       │   │   └── pseudo_atoms.def
│       │   └── O-Mg-O_morse_newFF_two
│       │       ├── force_field.def
│       │       ├── force_field_mixing_rules.def
│       │       └── pseudo_atoms.def
│       └── simulation.input
├── 5. Machine Learning
│   ├── CGCNN
│   │   ├── COF_list.csv
│   │   ├── README.md
│   │   ├── data
│   │   │   └── cif
│   │   │       ├── 21320N2.cif
│   │   │       ├── 21320N2_1.0_0.5.cif
│   │   │       ├── 21320N2_1.0_1.0.cif
│   │   │       ├── 21320N2_2.0_0.5.cif
│   │   │       ├── 21320N2_2.0_1.0.cif
│   │   │       ├── atom_init.json
│   │   │       ├── npy_gL
│   │   │       │   ├── 05001N2.npy
│   │   │       │   ├── ...
│   │   │       │   └── 21320N2.npy
│   │   │       ├── npy_wt
│   │   │       │   ├── 05001N2.npy
│   │   │       │   ├── ...
│   │   │       │   └── 21320N2.npy
│   │   │       ├── pkl
│   │   │       │   ├── 05001N2.pkl
│   │   │       │   ├── ...
│   │   │       │   └── 21320N2.pkl
│   │   │       └── pld.txt
│   │   ├── data.py
│   │   ├── model
│   │   │   ├── CGCNN_data.py
│   │   │   ├── CGCNN_model.py
│   │   │   └── CGCNN_run.py
│   │   ├── result
│   │   │   ├── gL
│   │   │   │   ├── loss_mae_train.txt
│   │   │   │   ├── loss_mae_valid.txt
│   │   │   │   ├── normalizer.pkl
│   │   │   │   ├── test.txt
│   │   │   │   ├── train.txt
│   │   │   │   └── val.txt
│   │   │   └── wt
│   │   │       ├── loss_mae_train.txt
│   │   │       ├── loss_mae_valid.txt
│   │   │       ├── normalizer.pkl
│   │   │       ├── test.txt
│   │   │       ├── train.txt
│   │   │       └── val.txt
│   │   ├── run_COF.py
│   │   ├── uptake_g.csv
│   │   └── uptake_w.csv
│   └── tree-based
│       ├── data.py
│       ├── dataset_gL.csv
│       ├── dataset_wt.csv
│       ├── dt_gL
│       │   ├── dt_gL.pkl
│       │   └── dt_gL.xlsx
│       ├── dt_wt
│       │   ├── dt_wt.pkl
│       │   └── dt_wt.xlsx
│       ├── gbr_gL
│       │   ├── gbr_gL.pkl
│       │   └── gbr_gL.xlsx
│       ├── gbr_wt
│       │   ├── gbr_wt.pkl
│       │   └── gbr_wt.xlsx
│       ├── model.py
│       ├── rf_gL
│       │   └── rf_gL.xlsx
│       ├── rf_wt
│       │   └── rf_wt.xlsx
│       ├── run.py
│       └── train.out
├── 6. Geometry Optimization of Functionalized COFs
│   ├── CHGNET
│   │   ├── 07010N3_ddec_two_all_chgnet.cif
│   │   └── run_chgnet.py
│   └── VASP
│       ├── optimization
│       │   ├── 07010N3_ddec_single_all
│       │   │   ├── 07010N3_ddec_single_all.cif
│       │   │   ├── calculation
│       │   │   │   ├── CHG
│       │   │   │   ├── ...
│       │   │   │   └── vasprun.xml
│       │   │   └── vasp_set.py
│       │   ├── 07010N3_ddec_two_all
│       │   │   ├── 07010N3_ddec_two_all.cif
│       │   │   ├── calculation
│       │   │   │   ├── CHG
│       │   │   │   ├── ...
│       │   │   │   └── vasprun.xml
│       │   │   └── vasp_set.py
│       │   ├── 21320N2_ddec_single_all
│       │   │   ├── 21320N2_ddec_single_all.cif
│       │   │   ├── calculation
│       │   │   │   ├── CHG
│       │   │   │   ├── ...
│       │   │   │   └── vasprun.xml
│       │   │   └── vasp_set.py
│       │   └── 21320N2_ddec_two_all
│       │       ├── 21320N2_ddec_two_all.cif
│       │       ├── calculation
│       │       │   ├── CHG
│       │       │   ├── ...
│       │       │   └── vasprun.xml
│       │       └── vasp_set.py
│       └── vasp_set.py
└── tree.txt


```
