# MLACDS
Machine Learning for the Acceleration of Colloid Diffusion Simulations

Final Year Project for TU880 Physics with Data Science Course in TU Dublin.

### Data
The data folder contains files with snapshots of diffusion data of particles in a 3d box. The naming convention is as follows:
p(some fraction)_N(some number) - volume percentage with N particles

e.g. p0.1_N54 - 0.1 percent of the box's volume is filled by 54 particles.

The relevant data in the files are stored as (seed).str and (seed).dat these contain particle xyz positions and their diffusion coefficients (Ds) respectively. The files contain a seed in their name which link the particle position file to the diffusion coefficient file.

The 4th 5th and 6th columns in the particle position files e.g. 1676475409.str contain the x y and z positions. The related diffusion coefficient file would be Ds_1676475409.dat. Notice how there are N_particles * 3 rows. This is because the first 3 rows are the Ds_x Ds_Y and Ds_z of the first particle, the next three for the second, etc. 


### MLACDS.py
This is a file containing useful functions for extraction of data in the format described above as well as functions to extract features such as nearest neighbours, local volume fraction bond order etc. (to be used for machine learning). Also will contain graphing functions for diffusion data.
