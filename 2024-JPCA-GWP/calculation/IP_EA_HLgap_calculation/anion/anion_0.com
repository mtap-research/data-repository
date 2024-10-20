%oldchk=mol_0.chk
%chk=anion_0.chk
%nprocs=16
%cpu=0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30
%mem=120GB
# b3lyp/6-31g(d,p) geom=check

test

-1 2 

