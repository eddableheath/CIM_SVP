# Ballistic Spreading Check
# Author: Edmund Dable-Heath
"""
    The original lattice planar graph CTQW paper was only interested in 2 dimensional analysis. Whilst not explicitly
    stated the key reason this appeared to be the case was for computational ease as simulating the quantum walk for
    much higher dimensions becomes infeasible. Their work implies that a CTQW on a square PLG in 2 dimensions, Z^2,
    exhibits ballistic spreading, i.e. var(Jt) >= (Jt)^2. This feels like it should extend to higher dimension, but
    does it?

    An analytic proof of this would be good as well, but some numerical simulations would be rather easy to achieve so
    those can be run first.
"""

import numpy as np
import CTQW as qw
import graph_functions as gf
import math
import os


def compute_variance(dimension, graph_bounds, gamma):
    """
        Compute the variance of the quantum walk distribution on a square PLG for a given dimension, graph size and
        ratio of propagation time to walker mass (gamma).
    :param dimension: dimension of the lattice, int
    :param graph_bounds: bounds of the graph, int
    :param gamma: ratio of propagation time and walker mass, float
    :return: variance of the output distribution over the PLG.
    """
    _, adj_mat = gf.generic_adjacency_matrix(graph_bounds, dimension)
    dist = qw.prop_comp(adj_mat, gamma)[(adj_mat.shape[0]-1)//2]
    return np.var(dist)


def compute_graph_bounds(dimension, total_points):
    """
        Compute the graph bounds to compute over based on the total number of points in the lattice. The total number
        of points will be based on the memory constraints available.
    :param dimension: dimension of the lattice, int
    :param total_points: total points in lattice/graph, int
    :return: bounds on the lattice, int
    """
    return math.floor(total_points ** (1 / float(dimension)) // 2)


def var_experiment(pars):
    gamma_vals = np.linspace(
        0,
        pars['max_gamma'],
        pars['max_gamma']*pars['gamma_granularity']
    )
    return {
        dim: {
            gamma: compute_variance(dim, compute_graph_bounds(dim, pars['max_points']), gamma)
            for gamma in gamma_vals
        }
        for dim in range(pars['dimension_range'][0], pars['dimension_range'][1]+1)
    }


def curve(gamma, a, p):
    return a * gamma**p


# Run
if __name__ == "__main__":
    var_pars = {
        'max_points': 68**4,
        'max_gamma': 5,
        'gamma_granularity': 5,
        'dimension_range': [2, 7]
    }

    results = var_experiment(var_pars)

    for dim in results.keys():
        np.savetxt('spreading_results/'+str(dim)+'.csv',
                   np.array([[gamma, results[dim][gamma]] for gamma in results[dim].keys()]),
                   delimiter=',')




