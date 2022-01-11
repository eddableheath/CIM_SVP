# Main File for the CTQW lattice SVP simulation
# Author: Edmund Dable-Heath
# The main file for putting together the entire simulation


import numpy as np
from copy import copy
import graph_functions as gf
import CTQW as qw
from auxillary_functions import metropolis_filter, lattice_gaussian
import time
import gc
import multiprocessing as mp
import basic_config as config
import os
import klein_sampler as ks
import math


def prop_coord(lattice_basis, propagation_time, hopping_amplitude, walk_dimension=None):
    """
        Generating the coordinates, propagator and adjacency matrix for a particular run as these are all constant for a
        given propagation time and lattice basis.
    :param lattice_basis: number theoretic lattice basis, fully defines the state space.
    :param propagation_time: propagation time for this case, for the propagator computation
    :param hopping_amplitude: hopping amplitude per unit time for jumping from one node to a neighbouring node
    :param walk_dimension: for keeping track of the dimension of the walk when working with sub dims of the lattice
    :return: tuple of propagator and coordinates
    """
    if walk_dimension == 1:
        coordinates, adj_matrix = gf.adj_mat_1d(lattice_basis)
    else:
        coordinates, adj_matrix = gf.adjacency_matrix(lattice_basis, walk_dimension)
    prop = qw.prop_comp(adj_matrix, propagation_time, hopping_amplitude)
    return prop, coordinates


def main_sim(lattice_basis, propagator, coordinates, chain_length, iterator, file_path, seed):
    """
        Main simulation of the quantum walk metropolis algorithm for sampling from lattices
    :param lattice_basis: basis for the number theoretic lattice
    :param propagator: already computed propagator for the quantum sim for a certain time
    :param coordinates: coordinates for this system
    :param chain_length: how many steps in the resulting Markov chain are we getting
    :param iterator: iterable keeping track of how many repeats of this run there have been.
    :param file_path: file path for storing the final
    :param seed: seed for the randomness.
    :return: Markov chain of points from the Lattice that should eventually converge
    """
    gc.collect()
    markov_chain = []
    lattice_dimension = lattice_basis.shape[0]

    # Pick initial position using klein sampler
    initial_latt_state = ks.klein_sampler(lattice_basis, 64,
                                          standard_deviation=1, seed=np.random.default_rng(int(12345*iterator)))
    initial_int_state = np.linalg.solve(lattice_basis.T, initial_latt_state).tolist()
    integer_bounds = math.ceil(lattice_dimension*math.log2(lattice_dimension) + math.log2(np.linalg.det(lattice_basis)))
    for i in range(len(initial_int_state)):
        if abs(initial_int_state[i]) > integer_bounds:
            initial_int_state[i] = np.sign(initial_int_state[i]) * (integer_bounds-1)
    initial_state = (initial_int_state, coordinates.index(list(map(int, initial_int_state))), initial_latt_state)
    markov_chain.append(initial_state)

    # Run quantum walk for new state
    for i in range(chain_length):
        proposal_state = qw.ctqw(propagator, markov_chain[-1][1])
        proposal_state = (np.asarray(coordinates[proposal_state]), proposal_state,
                          np.dot(lattice_basis.T, np.asarray(coordinates[proposal_state])))
        if np.linalg.norm(proposal_state[2]) == 0:
            markov_chain.append(copy(markov_chain[-1]))
        else:
            if metropolis_filter(markov_chain[-1], proposal_state, lattice_basis, lattice_gaussian, propagator):
                markov_chain.append(copy(proposal_state))
            else:
                markov_chain.append(copy(markov_chain[-1]))
    mc = [state[0] for state in markov_chain]
    np.savetxt(file_path+'/'+str(iterator)+'.csv', mc, delimiter=',')
    return 1


if __name__ == "__main__":

    # Run experiment here
    latt_basis = config.lattice_basis

    for time in config.test_times:
        path = 'quantum_results/t=' + str(time)
        try:
            os.mkdir(path)  # make directory for these results
        except FileExistsError:
            pass
        rng = np.random.default_rng(int(time*123456))

        pool = mp.Pool(config.cores)

        prop_spec, coords = prop_coord(latt_basis, time, config.hopping_parameter)

        iterables = range(config.number_of_runs)

        # for i in iterables:
        #     main_sim(latt_basis, prop_spec, coords,
        #              config.max_run_length, i, path)

        [pool.apply(main_sim,
                    args=(latt_basis,
                          prop_spec,
                          coords,
                          config.max_run_length,
                          i,
                          path,
                          rng))
         for i in iterables]

        pool.close()
        pool.join()



