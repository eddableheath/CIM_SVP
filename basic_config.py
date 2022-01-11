# Config file for CTQW run
# Author: Edmund Dable-Heath
# The config file for the basic continuous time quantum walk experiment.

import numpy as np
import math

# Multiprocessing parameters
cores = 4

# Lattice paramaters ---------------------------------
dimension = 2
lattice_type = 'hnf'
lattice_num = 0
lattice_basis = np.genfromtxt('Lattices/'+str(dimension)+'/'+str(lattice_num)+'/'+lattice_type+'.csv',
                              delimiter=',',
                              dtype=None)

# Model parameters -----------------------------------
particle_mass = 1
hopping_parameter = 1 / particle_mass       # set a = hbar = 1

max_time = 5
time_split = int(math.ceil(max_time / 0.5))
test_times = np.linspace(0.5, max_time, num=time_split)

max_run_length = 10000
number_of_runs = 100
