# Config file for CTQW moving run
# Author: Edmund Dable-Heath
"""
    The config file for the basic continuous time quantum walk experiment.
"""

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

gamma = 3.5

number_of_runs = 1000
