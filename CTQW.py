# Continuous Time Quantum Walk Simulation
# Author: Edmund Dable-Heath
# This is going to be a very rudimentary simulation of a continuous time quantum walk for an arbitrary graph. The idea
# to have as inputs an adjacency matrix, a permeation time and a hopping amplitude. This should fully describe the
# entire simulation. It won't be efficient, I'll work on that afterward.


import numpy as np
from scipy.linalg import expm
import graph_functions as gf


def prop_comp(adjacency_matrix, propagation_time, hopping_amplitude):
    """
        Computing the propagator unitary matrix for the simulation, considering a fixed prop time and hopping amplitude.
    :param adjacency_matrix: adjacency matrix of the underlying graph
    :param propagation_time: propagation time for each quantum step
    :param hopping_amplitude: hopping amplitude per unit time for jumping from one node to a neighbouring node
    :return: np array of unitary matrix propagtor
    """
    graph_laplacian = adjacency_matrix - np.diag(np.sum(adjacency_matrix, axis=0))
    hamiltonian = -hopping_amplitude * graph_laplacian
    return expm(1j * hamiltonian * propagation_time)


def ctqw(propagator, initial_position, seed=None):
    """
        continuous time quantum walk on arbitrary graph
    :param propagator: unitary time evolution operator for fixed graph, time, and hopping amplitude.
    :param initial_position: the initial position of the walk, if not given will be uniformly randomly picked.
    :param seed: seed for the randomness
    :return: end state of the walk, ready to be measured and 'collapse' into a single state.
    """
    if not seed:
        seed = np.random.default_rng()
    return seed.choice(propagator.shape[0], p=np.absolute(propagator[initial_position])**2)


if __name__ == "__main__":
    basis = np.array([[32, 0, 0],
                      [1, 1, 0],
                      [1, 0, 1]])
    coord, adj = gf.adj_mat_1d(basis)
    walk = ctqw(prop_comp(adj, 10, 1), 0)
    print(walk)
    print(coord[walk])
