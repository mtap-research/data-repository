%chk=mol_0.chk
%nprocs=16
%cpu=0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30
%mem=120GB
# opt=calcfc freq b3lyp/6-31g(d,p) 

test

0 1
C          0.91769        0.02519        0.04112
F          0.46423        0.85245       -0.93899
Cl         0.31568       -1.61480       -0.27240
Cl         0.31569        0.60968        1.60517
Cl         2.69246        0.03945        0.02423

