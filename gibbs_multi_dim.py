# Sub routine quantum gibbs multi dim version
# Author: Edmund Dable-Heath
# This version is to make up for the fact that the space scaling is particularly poor, so instead the new direction is
# to split the dimension into either d/2 two dimensional walks or floor(d/2)-1 two dimension and one three dimensional
# walk depending on whether the dimension is even or not. Also in the plan is to refactor this using object oriented
# experiment design.
#
# The scaling of the size of the propagator means that running 3 dimensional walks is proving difficult, probably worth
# resorting to a one dimensional walk for the odd dimensions on top of the 2 dimensional ones.
#
# Have now changed it to 1 dimensional version, appears to work much better. Need to double check that it's giving me
# what I want though, should write some unit tests to check it.

import gc
import os
import multiprocessing as mp
import numpy as np
import auxillary_functions as af
import CTQW as qw
import quantum_main as qm
import gc
import klein_sampler as ks
import math
from copy import copy
import met_gibbs_config as config


def main(lattice_basis, propagation_time, hopping_amplitude, chain_length, iterator, file_path, random_scan=False):
    """
        Main simulation for the metropolis within in Gibbs model
    :param lattice_basis: lattice basis, ndarray
    :param propagation_time: propagation time for quantum walk, float
    :param hopping_amplitude: hopping amplitude per unit time for jumping from one node to a neighbouring node
    :param chain_length: number of points in the final markove chain
    :param iterator: iterable for keeping track of how many repeats of this run there have been.
    :param file_path: file path for writing results
    :param random_scan: set the scanning mode for the algorithm to be random.
        # Todo: code up choice of scanning mode
    :return: Markove chain of points from the lattice that should be picking short vectors.
    """
    gc.collect()
    markov_chain = []
    lattice_dimension = lattice_basis.shape[0]

    # Computing propagators and coordinates
    # NOTE: commented out section for previous splitting scheme.
    # if lattice_dimension%2 == 0:
    #     prop2, coord2 = qm.prop_coord(lattice_basis, propagation_time, hopping_amplitude, walk_dimension=2)
    # else:
    #     prop2, coord2 = qm.prop_coord(lattice_basis, propagation_time, hopping_amplitude, walk_dimension=2)
    #     prop3, coord3 = qm.prop_coord(lattice_basis, propagation_time, hopping_amplitude, walk_dimension=3)
    # Actual computation.
    if lattice_dimension%2 == 0:
        prop2, coord2 = qm.prop_coord(lattice_basis, propagation_time, hopping_amplitude, walk_dimension=2)
    else:
        prop2, coord2 = qm.prop_coord(lattice_basis, propagation_time, hopping_amplitude, walk_dimension=2)
        prop1, coord1 = qm.prop_coord(lattice_basis, propagation_time, hopping_amplitude, walk_dimension=1)

    # Sampling initial point using Klein sampler
    initial_latt_state = ks.klein_sampler(lattice_basis, 64,
                                          standard_deviation=1, seed=np.random.default_rng(int(1234*iterator)))
    initial_int_state = np.linalg.solve(lattice_basis.T, initial_latt_state).tolist()
    integer_bounds = math.ceil(lattice_dimension*math.log2(lattice_dimension) + math.log2(np.linalg.det(lattice_basis)))
    for i in range(len(initial_int_state)):
        if abs(initial_int_state[i]) > integer_bounds:
            initial_int_state[i] = np.sign(initial_int_state[i]) * (integer_bounds - 1)
    initial_state = (initial_int_state, np.dot(lattice_basis, initial_int_state))
    markov_chain.append(initial_state)

    for i in range(chain_length):
        # Run quantum walk subroutine on subdims for new state gen
        subdim_splits = af.split_dim(lattice_dimension, random_scan)
        for subdim in subdim_splits:
            current_state = copy(markov_chain[-1][0])
            if type(subdim) == tuple:
                subdim_vec = np.take(current_state, list(subdim))
            else:
                subdim_vec = current_state[subdim]
            proposal_state = copy(current_state)
            if type(subdim) == tuple:
                proposal_subdim = coord2[qw.ctqw(prop2, coord2.index(list(map(int, subdim_vec))))]
                for i in range(len(proposal_subdim)):
                    proposal_state[subdim[i]] = proposal_subdim[i]
            else:
                proposal_subdim = coord1[qw.ctqw(prop1, coord1.index(subdim_vec))]
                proposal_state[subdim] = proposal_subdim

            if np.linalg.norm(proposal_state) == 0:
                markov_chain.append(copy(markov_chain[-1]))
            else:
                if af.metropolis_filter_simple(current_state, proposal_state, lattice_basis):
                    markov_chain.append((copy(proposal_state), np.dot(lattice_basis.T, copy(proposal_state))))
                else:
                    markov_chain.append(markov_chain[-1])

    mc = [state[1] for state in markov_chain]
    np.savetxt(file_path+'/'+str(iterator)+'.csv', mc, delimiter=',')
    return 1


if __name__ == "__main__":

    # Run experiment here
    latt_basis = config.lattice_basis

    for time in config.test_times:
        path = 'quantum_results/t=' +str(time)
        try:
            os.mkdir(path) # make directory for these results
        except FileExistsError:
            pass
        rng = np.random.default_rng(int(time*123456))

        pool = mp.Pool(config.cores)

        iterables = range(config.number_of_runs)

        [pool.apply(main,
                    args=(latt_basis,
                          time,
                          config.hopping_parameter,
                          config.max_run_length,
                          i,
                          path,
                          True))
         for i in iterables]

        pool.close()
        pool.join()