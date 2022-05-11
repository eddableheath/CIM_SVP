# Config for CIM SVP experiment
# Author: Edmund Dable-Heath
"""
    Config file for the CIM SVP experiment.
"""

import numpy as np

# Lattice parameters ---------------------------------------------------------------
dimension = 2
latt_numbers = 0
latt_basis = np.array([[-1, 0, -1, -3, 0],
                       [7, 0, -11, -7, 9],
                       [-6, -1, 0, -9, -5],
                       [-2, 0, -3, -8, 0],
                       [5, 1, -7, -5, 6]])  # read the basis from file
latt_int_bounds = None  # Set these from Chris's work
sitek = 6               # compute this from latt_int_bounds
qudit_mapping = 'poly'  # ham, poly or bin

# CIM parameters -------------------------------------------------------------------
control_system = 'CAC'
couplings_matrix = np.random.randn(5, 5)
for i in range(couplings_matrix.shape[0]):
    couplings_matrix[i, i] = 0.
pump_start = -2.
pump_end = 0.
tamp_start = 1.
tamp_end = 2.
beta = 0.2
c = 1.
k = 1.


# limiting functions
def CAC_lim_func(tamp):
    return (3/2) * np.sqrt(tamp)


def CFC_lim_func(tamp):
    return 3/2


def SFC_lim_func(tamp):
    return None


if control_system == 'CAC':
    limit_func = CAC_lim_func
elif control_system == 'CFC':
    limit_func = CFC_lim_func
elif control_system == 'SFC':
    limit_func = SFC_lim_func

nl_func = 1

# Experimentation parameters -------------------------------------------------------
time_diff = 0.001
iters = 32000
var_iters = 28800
stat_iters = 3200
repeats = 10
save_trajectory = False
write_path = 'results'

# Multiprocessing parameters -------------------------------------------------------
cores = 4

# Pars dict ------------------------------------------------------------------------
pars = {
    'basis': latt_basis,
    'sitek': sitek,
    'qudit_mapping': qudit_mapping,
    'control_sys': control_system,
    'couplings': couplings_matrix,
    'pump_start': pump_start,
    'pump_end': pump_end,
    'tamp_start': tamp_start,
    'tamp_end': tamp_end,
    'beta': beta,
    'k': k,
    'c': c,
    'limit_func': limit_func,
    'nl_func': nl_func,
    'time_diff': time_diff,
    'iters': iters,
    'var_iters': var_iters,
    'stat_iters': stat_iters,
    'repeats': repeats,
    'save_traj': save_trajectory,
    'cores': cores,
    'path': write_path
}
