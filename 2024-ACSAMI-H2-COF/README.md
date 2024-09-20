# Computational Exploration of Adsorption-based Hydrogen Storage in Mg-alkoxide Functionalized Covalent-Organic Frameworks (COFs): Force-field and Machine Learning Models

* Publication: (to-be-updated)
  
* DOI: to-be-updated

This repository includes all the input files and some example output files for GCMC simulation, computational structural models of functionalized COFs, and the raw data for plotting the figures in the manuscript (simulation-related sections).

This repository was created by Yu Chen.

## Files Directory:
```
.
├── 1. Functionalized COFs
│   ├── README.txt
│   ├── func_COF_label.zip
│   └── func_COF_unlabel.zip
├── 2. FF Fitting
│   ├── E_LJ_calc.ipynb
│   ├── FF_LJ_coulomb_params.csv
│   ├── FF_fitting-arranged-ver2.ipynb
│   ├── LJ-Coulomb_E_bind.csv
│   ├── MP2_E_bind.xlsx
│   └── README.md
├── 3. H2 - GCMC
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
├── 4. Machine Learning
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
│   │   │   │   ├── checkpoints
│   │   │   │   │   └── model.pth
│   │   │   │   ├── loss_mae_train.txt
│   │   │   │   ├── loss_mae_valid.txt
│   │   │   │   ├── normalizer.pkl
│   │   │   │   ├── test.txt
│   │   │   │   ├── train.txt
│   │   │   │   └── val.txt
│   │   │   └── wt
│   │   │       ├── checkpoints
│   │   │       │   └── model.pth
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
│       ├── pip_list
│       ├── rf_gL
│       │   └── rf_gL.xlsx
│       ├── rf_wt
│       │   └── rf_wt.xlsx
│       ├── run.py
│       └── train.out
└── tree.txt


```
