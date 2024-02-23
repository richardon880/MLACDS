
#### 

#lammps-daily -in lan_poly_1_size.in -log lan_L$1.log -var L $1 -var outfile $2 -var SEED2 $3 

#lammps-daily -in lan_poly_N_size.in -log lan_L$1.log -var L $1 -var outfile $2
lmp -in lan_poly_N_size.in -log lan_L$1.log -var L $1 -var outfile $2

