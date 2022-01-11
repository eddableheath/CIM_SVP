# Main classical algorithm implementation
# Author: Edmund Dable-Heath
# The main implementation of the discontinous hamiltonian markov chain monte carlo algorithm over the integers.
import math

import numpy as np
import auxillary_functions as fn
from copy import copy


def main_dhmc(basis, initial_point, mass_matrix_diag, step_size, total_steps):
    """
    main function for running the discontinuous hamiltonian MCMC algorithm over integers for the lattice
    :param basis: lattice basis, n*n numpy array
    :param initial_point: initial point (picked from some starting distribution)
    :param mass_matrix_diag: mass matrix, assumed to be diagonal so currently is a vector.
    :param step_size: the size inbetween each step, to be adjusted
    :param total_steps: total steps in the chain
    :return: a series of points in the lattice, sampled via the integers
    """
    dimension = basis.shape[0]

    # 'infinite zero point energy' - simply arbitrarily high to avoid the algorithm getting stuck at zero and turn the
    # shortest vector into the global minimum
    zero_point_energy = abs(np.linalg.det(basis))

    # initial momentum based on the mass matrix
    initial_momentum = np.random.laplace(scale=mass_matrix_diag)

    # permutation of indices
    iterables = np.random.permutation(dimension)

    # results
    results = np.zeros((total_steps, dimension))
    results[0] = initial_point

    lattice_results = np.zeros((total_steps, dimension))
    lattice_results[0] = np.dot(basis.T, initial_point)

    # inr_mediary_results = np.zeros((total_steps*dimension, dimension))
    # inter_mediary_results[0] = initial_point

    for i in range(1, total_steps):
        working_point = copy(results[i - 1])
        iterable = iterables[i % dimension]
        working_point[iterable] = working_point[iterable] + \
                                  step_size*(1 / mass_matrix_diag[iterable])*np.sign(initial_momentum)[iterable]
        if (np.abs(working_point) > fn.range_calc(basis)).any():
            initial_momentum[iterable] = -initial_momentum[iterable]
            results[i] = results[i - 1]
            lattice_results[i] = lattice_results[i - 1]
            continue
        new_potential = fn.potential_energy(basis, working_point, zero_point_energy)
        old_potential = fn.potential_energy(basis, results[i-1], zero_point_energy)
        potential_difference = new_potential - old_potential
        if (1/mass_matrix_diag[iterable])*abs(initial_momentum[iterable]) > potential_difference:
            results[i] = working_point
            initial_momentum[iterable] -= np.sign(initial_momentum[iterable]) * \
                                          mass_matrix_diag[iterable] * potential_difference
            lattice_results[i] = np.dot(basis.T, np.around(working_point))
        else:
            initial_momentum[iterable] = -initial_momentum[iterable]
            results[i] = results[i-1]
            lattice_results[i] = lattice_results[i-1]
    return results, lattice_results


if __name__ == "__main__":
    lattice = np.array([[2, -1, 1],
                        [2, -1, -3],
                        [-2, -3, -2]])
    init_point = fn.non_zero_init_point(lattice)
    mass_matrix = np.array([0.7, 0.7, 0.7])
    step = 0.5
    ints, latts = main_dhmc(lattice, init_point, mass_matrix, step, 10000)
    print(latts)
    print(fn.find_min_vector(latts))
