# Moving the Graph Method
# Author: Edmund Dable-Heath
"""
    In this version of the quantum walk algorithm instead of having a fixed graph and the quantum walk taking steps
    around it the quantum walk will always walk from the centre of the graph with the graph being embedded into Z^n
    around the current point. This will allow for a much computationally simpler approach only requiring a single row of
    the propagator to be stored for the most part. We know the bounds on Z^n that will contain the shortest vector for a
    particular lattice so if the graph around a new point is going to step over the bounds we walk over a temporary
    reduced graph that is restricted by the boundary.
"""

import numpy as np
import graph_functions as gf
import CTQW as qf
import networkx as nx
import auxillary_functions as af
from copy import copy
import math
import os


def gen_generic_prob_density(dimension, graph_bounds, gamma):
    """
        For a generic walk on a square planar lattice graph in a specified dimension or specified range and propagation
        time output the probability density generated from the walker starting at the central node.
    :param dimension: dimension of the graph to be searched over.
    :param graph_bounds: bounds of the graph to be searched over.
    :param gamma: single parameter for this model, the ration of the propagation time to the mass of the walker.
    :return: (m,) real ndarray representing the probability.
    """
    coords, adj_mat = gf.generic_adjacency_matrix(graph_bounds, dimension)
    return coords, np.absolute(qf.prop_comp(adj_mat, gamma)[(adj_mat.shape[0]-1)//2])**2


def compute_graph_bounds(lattice_basis, total_points):
    """
        Compute the range that the graph walk should be to be reasonably computable. Given a total number of points it
        will give bounds that have either approximately those number of points spread over the correct dimensions or
        the full integer bounds.
    :param lattice_basis: lattice basis for computation of the dimension and integer bounds, int (m,m)-ndarray
    :param total_points: total points to be included, int
    :return: maximum bounds for the graph in each dimension, int
    """
    dimension = lattice_basis.shape[0]
    integer_bounds = math.ceil(dimension * math.log2(dimension) + math.log2(np.linalg.norm(lattice_basis)))
    bound = math.floor(total_points ** (1/float(dimension)) / 2)
    if bound > integer_bounds:
        return integer_bounds
    else:
        return bound


class WalkExperiment:

    """
        Class for the walk experiment.

        Cost function has two choices:
            - 'gauss' selects the lattice gaussian as the target distribution.
            - 'log' selects |log(||Bx||)| as the cost function.
    """

    def __init__(
            self, lattice_basis, prop_density, graph_coords, graph_bounds, markov_iters, markov_comm,
            cost_choice='gauss'
    ):
        self.basis = lattice_basis
        self.dimension = self.basis.shape[0]
        self.int_lattice_bounds = self.dimension * np.log2(self.dimension) + np.log2(np.linalg.det(self.basis))

        self.prob_density = prop_density
        self.graph_coords = graph_coords
        self.graph_bounds = graph_bounds

        self.current_integer_vector = np.random.randint(-self.graph_bounds, self.graph_bounds+1, self.dimension)
        self.markov_chain = [copy(self.current_integer_vector)]
        self.lattice_markov_chain = [self.basis.T @ copy(self.current_integer_vector)]
        self.markov_iters = markov_iters
        self.markov_comm = markov_comm
        self.markov_cost_choice = cost_choice

    def update_state(self):
        proposal_int_state = self.current_integer_vector + \
                             np.asarray(self.graph_coords[np.random.choice([i
                                                                            for i in range(self.prob_density.shape[0])],
                                                                           p=self.prob_density.tolist())])
        while np.any(proposal_int_state > self.int_lattice_bounds):
            # print(proposal_int_state)
            proposal_int_state = self.current_integer_vector + \
                                 np.asarray(self.graph_coords[np.random.choice([i for i
                                                                                in range(self.prob_density.shape[0])],
                                                                               p=self.prob_density.tolist())])
        if self.markov_cost_choice == 'gauss':
            if af.metropolis_filter_simple(self.current_integer_vector, proposal_int_state, self.basis):
                self.current_integer_vector = proposal_int_state
        elif self.markov_cost_choice == 'log':
            if af.metropolis_filter_log_cost(self.current_integer_vector, proposal_int_state, self.basis):
                self.current_integer_vector = proposal_int_state
        self.markov_chain.append(copy(self.current_integer_vector))
        self.lattice_markov_chain.append(self.basis.T @ copy(self.current_integer_vector))

    def run(self):
        for i in range(self.markov_iters):
            self.update_state()
            # print(self.markov_chain)

    # Uncomment and finish up below for alt graph searching method rather than pruning method
    # def compute_overspill(self):
    #     overspill_amounts = np.zeros_like(self.current_integer_vector)
    #     for i in range(self.current_integer_vector.shape[0]):
    #         if abs(self.current_integer_vector[i]) + self.graph_bounds > self.int_lattice_bounds:
    #            overspill_amounts[i] = np.sign(self.current_integer_vector[i]) * (abs(self.current_integer_vector[i]) +
    #                                                                               self.graph_bounds -
    #                                                                               self.int_lattice_bounds)
    #     return overspill_amounts

    # def compute_alt_prob_density(self, overspill):
    #     grid_graph = nx.grid_graph(dim=[2*self.graph_bounds + 1 - overspill[i]
    #                                     for i in range(self.dimension)])
    #     coords = [[nodes[index] - self.graph_bounds]]

    # def update_state(self):
    #     overspill = self.compute_overspill()
    #     if sum(overspill) == 0:
    #         dist = self.prob_density
    #         coords = self.graph_coords
    #     else:
    #         x=1
    #     self.current_integer_vector = self.current_integer_vector + np.asarray(coords[np.random.choice(dist.shape[0],
    #                                                                                                    p=dist)])


# Testing
if __name__ == "__main__":
    # basis = np.array([[0, -1, -1, 1, 0],
    #                   [1, -1, 0, 0, 1],
    #                   [-1., -1, 1, 0, 1],
    #                   [0, 1, -1, -1, 1],
    #                   [0, 1, 1, 3, 1]])
    for i in range(2, 8):
        os.mkdir('results/'+str(i))
        for j in range(32):
            os.mkdir('results/'+str(i)+'/'+str(j))
            for k in range(1000):
                print('current index:'+str(i)+str(j)+str(k))
                basis = np.genfromtxt('Lattices/'+str(i)+'/'+str(j)+'/lll.csv', delimiter=',', dtype=None)
                prop_pars = {
                    'dimension': basis.shape[0],
                    'graph_bounds': compute_graph_bounds(basis, 31**2),
                    'gamma': 3.5
                }
                coords, prob_density = gen_generic_prob_density(prop_pars['dimension'],
                                                                prop_pars['graph_bounds'],
                                                                prop_pars['gamma'])
                pars = {
                    'basis': basis,
                    'prob_density': prob_density,
                    'coords': coords,
                    'graph_bounds': prop_pars['graph_bounds'],
                    'markov_iters': 10000,
                    'markov_comm': 1e-7,
                    'cost_choice': 'gauss'
                }

                test_exp = WalkExperiment(
                    pars['basis'],
                    pars['prob_density'],
                    pars['coords'],
                    pars['graph_bounds'],
                    pars['markov_iters'],
                    pars['markov_comm'],
                    pars['cost_choice']
                )

                path = 'results/'+str(i)+'/'+str(j)+'/'+str(k)
                print('save path: '+path)
                os.mkdir(path)

                experiment = WalkExperiment(
                    pars['basis'],
                    pars['prob_density'],
                    pars['coords'],
                    pars['graph_bounds'],
                    pars['markov_iters'],
                    pars['markov_comm'],
                    pars['cost_choice']
                )
                experiment.run()
                np.savetxt(path+'/ints.csv', X=np.array(experiment.markov_chain), delimiter=',')
                np.savetxt(path+'/latts.csv', X=np.array(experiment.lattice_markov_chain), delimiter=',')
