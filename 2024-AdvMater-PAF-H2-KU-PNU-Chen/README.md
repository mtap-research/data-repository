# High Hydrogen Storage in Trigonal Prismatic Monomer-based Highly Porous Aromatic Frameworks

* Publication: Advanced Materials (Accepted)
  
* DOI: [10.1002/adma.202401739](https://doi.org/10.1002/adma.202401739)

This repository includes all the input files and some example output files for GCMC simulation, computational structural models of PAFs, and the raw data for plotting the figures in the manuscript (simulation-related sections).

This repository was created by Yu Chen.

## Files Directory:
```
ORGANIZED simulation files/
│
├── README.md
│
├── H2 - GCMC/
│   ├── density maps calculation/
│   │   ├── example output files - C_acs_opt-1bar-298K/
│   │   │   └── output_C_acs_opt_1.1.1_298.000000_0.data
│   │   └── input files/
│   │       ├── H2.def
│   │       ├── force_field.def
│   │       ├── force_field_mixing_rules.def
│   │       ├── pseudo_atoms.def
│   │       ├── simulation.input_density_maps
│   │       └── simulation.input_generate_grid
│   └── isotherms calculation/
│       ├── example output files - C_acs_opt-1bar-77K/
│       │   └── output_C_acs_opt_1.1.1_77.000000_1e+07.data
│       └── input files/
│           ├── H2.def
│           ├── force_field.def
│           ├── force_field_mixing_rules.def
│           ├── pseudo_atoms.def
│           └── simulation.input
│
├── N2 - GCMC/
│   ├── example output files - C_acs_opt-1bar-77K/
│   │   └── output_C_acs_dftb_opt_1.1.1_77.000000_100000.data
│   └── input files/
│       ├── N2.def
│       ├── force_field_mixing_rules.def
│       ├── pseudo_atoms.def
│       └── simulation.input
│
├── PAF modeling/
│   ├── defective models - optimized/
│   │   ├── C_acs_defect_4_opt.cif
│   │   ├── C_acs_defect_8_cluster_opt.cif
│   │   ├── Si_acs_defect_4_opt.cif
│   │   └── Si_acs_defect_8_cluster_opt.cif
│   ├── idealized crystalline models - optimized/
│   │   ├── C_acs_opt.cif
│   │   ├── C_bcs_opt.cif
│   │   ├── C_crs_opt.cif
│   │   ├── C_lcy_opt.cif
│   │   ├── C_pcu_opt.cif
│   │   ├── Si_acs_opt.cif
│   │   ├── Si_bcs_opt.cif
│   │   ├── Si_crs_opt.cif
│   │   ├── Si_lcy_opt.cif
│   │   └── Si_pcu_opt.cif
│   └── README.md
│
└── raw_datasheet_figures.xlsx

```
