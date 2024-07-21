# Data preparation

## Introduction

- First, calculate single isotherms and thermodynamic parameters for Xe & Kr by [RASPA2](https://github.com/iRASPA/RASPA2) and physcial features by [Zeo++](http://zeoplusplus.org/)
- Next, fitting the isotherms using Single-site Langmuir model.
  - N = M $\dfrac{KP}{1+KP}$
    - N: uptake; M: saturation loading; K: Langmuir parameter; P: pressure.
- Final, Is the saturated loading estimated by the conventional method accurate enough? & Is the K calculated by the thermodynamic parameters accurate?
  - K = $\dfrac{K_H}{M}$
    - $K_H$: Hencry constan;
  - M = PV $\rho_l$
    - PV: pore volume; $\rho_l$: density of liquid gas.

## Installation of Dependency

- python 3.8.3
- numpy 1.21.4
- matplotlib 3.2.2
- scipy 1.6.3
- pandas 1.3.4

## GCMC simulation

There is an example simulation input file                                   
Force field parameters can be found in Supporting Information of Xe/Kr literature
