# CIM CAC config file
# Author: Edmund Dable-Heath
# All the necessary config in here to separate it from the other modules, easily accessible. This will call functions
# from lattice_to_Jij and serve the main run file. All necessary parameters for the run go in here, every other file is
# purely for running scripts.

import numpy as np
import lattice_to_ising as lj
import CIM_CAC_sim as sim

# Lattice parameters -------------------------------------------------------------------------------------
lattice_dim = 5
lattice_type = 'lll'
lattice_no = 0
# lattice_basis = np.genfromtxt('Lattices/'+str(lattice_dim)+'/'+str(lattice_no)+'/'+lattice_type+'.csv',
#                               delimiter=',', dtype=None)
lattice_basis = np.array([[3, 4],
                          [2, 3]])
latt_int_bounds = None
encoding = 'ham'
# max_norm_bound = lj.minkowski_energy(lattice_basis)**2
gramm = lattice_basis@lattice_basis.T
sitek = 4

# lattice interaction matrix
q_level_spins = False
if q_level_spins:
    lattice_interactions = lj.Jij_qary(lattice_basis)
else:
    if encoding == 'bin':
        lattice_interactions = lj.svp_isingcoeffs_bin(lattice_basis@lattice_basis.T, sitek)[0]
        # print(lattice_interactions)
    else: # defaults to Ham encoding
        lattice_interactions = lj.svp_isingcoeffs_ham(gramm, sitek)[0]

use_lattice_interactions = True

# print(lattice_interactions)

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
