# Implementation of the Klein Sampler
# Author: Edmund Dable-Heath
# Using Thomas Prest's algorithm

import numpy as np
import math
# from quantum_main import rng


def one_d_int_gauss(centre, standard_deviation, security_parameter, seed=None):
    """
    One dimensional discrete gaussian sampler over the integers.
    :param centre: centre of the distribution
    :param standard_deviation: standard deviation of the distribution
    :param security_parameter: security parameter for computing range of support.
    :param seed: seed for the randomness
    :return: an integer sample from the relevant distribution
    """
    if not seed:
        seed = np.random.default_rng()
    while(1):
        uniform_pick = seed.integers(centre-(standard_deviation*math.log2(security_parameter)),
                                     centre+(standard_deviation*math.log2(security_parameter)))
        rejection_bound = (1/(standard_deviation*math.sqrt(2*math.pi))) * np.exp(-0.5*(((uniform_pick-centre)/standard_deviation)**2))
        rejection_test = seed.uniform(0, 1)
        if rejection_test <= rejection_bound:
            return uniform_pick
        else:
            continue


def klein_sampler(lattice_basis,  security_parameter, standard_deviation=1, target=0, seed=None):
    """
    Klein Sampler Algorithm for sampling from a discrete gaussian over a lattice.
    :param lattice_basis: lattice basis
    :param security_parameter: security parameter for rejection sampling step
    :param standard_deviation: standard deviation of the gaussian
    :param target: centre of the distribution, assumed to be origin if not specified.
    :param seed: seed for the randomness
    :return: sample from discrete gaussian over the lattice.
    """
    dimension = lattice_basis.shape[0]
    gso_basis, _ = np.linalg.qr(lattice_basis)
    gso_basis = gso_basis.T
    if target == 0:
        target = np.zeros(dimension)
    update_vector = np.zeros(dimension)
    for i in range(dimension, 0, -1):
        i -= 1
        d_val = np.dot(target, gso_basis[i]) / np.linalg.norm(gso_basis[i])**2
        sigma_val = standard_deviation / np.linalg.norm(gso_basis[i])
        z_val = one_d_int_gauss(d_val, sigma_val, security_parameter, seed=seed)
        target -= z_val*lattice_basis[i]
        update_vector += z_val*lattice_basis[i]
    return update_vector


if __name__=="__main__":
    latt_basis = np.array([[32, 0],
                           [9, 1]])
    for i in range(20):
        latt_sample = klein_sampler(latt_basis, 64, standard_deviation=1, seed=np.random.default_rng(int(12346*2)))
        int_sample = np.linalg.solve(latt_basis.T, latt_sample)
        print(f'lattice_sample: {latt_sample}')
        print(f'integer_sample: {int_sample}')
        print('----------------------------------')