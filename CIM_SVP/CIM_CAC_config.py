# CIM CAC config file
# Author: Edmund Dable-Heath
# All the necessary config in here to separate it from the other modules, easily accessible. This will call functions
# from lattice_to_Jij and serve the main run file. All necessary parameters for the run go in here, every other file is
# purely for running scripts.

import numpy as np
import isingify as lj
import CIM_CAC_sim as sim
import os
import fnmatch

# Lattice parameters -------------------------------------------------------------------------------------
lattice_dim = 2
lattice_no = 2
lattice_basis = np.genfromtxt(f'lattices/{str(lattice_dim)}/{str(lattice_no)}_1.csv',
                              delimiter=',', dtype=None)
latt_int_bounds = None
encoding = 'poly'
# max_norm_bound = lj.minkowski_energy(lattice_basis)**2
gramm = lattice_basis@lattice_basis.T
sitek = 4 # Todo: Refactor to take into account some lattice bounds.

# lattice interaction matrix
if encoding == 'poly':
    lattice_interactions, identity_coeff = lj.poly_couplings(gramm, sitek)
elif encoding == 'bin':
    lattice_interactions, identity_coeff = lj.bin_couplings(lattice_basis@lattice_basis.T, sitek)
elif encoding == 'ham':
    lattice_interactions, identity_coeff = lj.ham_couplings(gramm, sitek)
print(lattice_basis)
print(lattice_interactions)
print(identity_coeff)
use_lattice_interactions = True

# Interactions -------------------------------------------------------------------------------------------
instance_set_path = "../InstanceSets/BENCHSKL/"
instance_file_dir = "BENCHSKL_800_100/"
inst_file_common_name = "BENCHSKL_800_100_"
instance_number = 0
problem_size = 12

if use_lattice_interactions:
    interactions = lattice_interactions
else:
    interactions = np.random.randint(1, 10, (6, 6))

# General CIM parameters ---------------------------------------------------------------------------------

# Ground energy - pre compute for large problem sizes?
ground_energy = 1.

# amplitude
tamp_start = 1.
tamp_raise = 2.

beta = 0.8

# pumpfield parameters
pumpfield = -2.0  # Note: this is equal to p-1 factor from paper
pf_raise = 2.0
pf_start = pumpfield
pf_end = pumpfield + pf_raise

# timestep
time_step = 0.125
time_splits = 3200
no_repetitions = 10

# results tolerances
res_tol = 1E-7
