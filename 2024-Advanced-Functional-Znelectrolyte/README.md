# Polyoxometalate Initiated In situ Conformal Coating of Multifunctional Hybrid Artificial Layers for High Performance Zinc Metal Anodes

* Publication: 02 September 2024
  
* DOI: https://doi.org/10.1002/adfm.202412577

This repository includes all the input files and some example output files for classical molecular dynamics simulation by LAMMPS. Simulation is for identify the effect of POMDOL layer on Zn anode by analyzing trajectory.
This repository was created by Haewon Kim.

## Files Directory:
```
.
├── 1. No-POMDOL-Zn-electrolyte
│   └── Lammps
│      ├── analysis
│      │     ├──data.after_wide_nvt
│      │     ├──rerun-msd.run
│      │     ├──rerun-rdf.run
│      │     ├──system.in.init
│      │     └──system.in.settings
│      ├── efield
│      │     ├──data.after_wide_350_nvt
│      │     ├──efield-nvt.run
│      │     ├──system.in.constraints
│      │     ├──system.in.init
│      │     └──system.in.settings
│      └── NVT
│            ├──nvt.run
│            ├──system.data
│            ├──system.in.constraints
│            ├──system.in.init
│            └──system.in.settings
│
└── 2.POMDOL-Zn-electrolyte
    └── Lammps
       ├── analysis
       │     ├──rerun-rdf
       │     └──tmp_zn_pomdol_msd.rdf
       ├── efield
       │     ├──data.after_efield_pomdol_2
       │     ├──data.after_wide_nvt_350_pomdol
       │     ├──efield-nvt.run
       │     ├──system.in.constraints
       │     ├──system.in.init
       │     └──system.in.settings
       └── NVT
             ├──data.after_wide_nvt_350_pomdol
       │     ├──data.after_wide_nvt_pomdol
             ├──nvt.run
             ├──system.in.constraints
             ├──system.in.init
             └──system.in.settings
```
