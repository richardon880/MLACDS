
##dir=/home/hender/Research/ILL/cons_R/V2/mono/0.1/
#dir=$1
dir=$(pwd)
#dirc=/home/hender/Research/ILL/Ly_Ando_new_sims/Lysate_v2/common/
dirc=/home/richard/Desktop/Dshort_code/common/

python $dirc/run_set_lammps.py  

python $dirc/run_gene_box.py $dir

python $dirc/cal_M_ewald.py $dir

python $dirc/calc_Rlub.py  $dir 

python $dirc/cal_average.py $dir $dirc

cat Ds_* > res.dat


