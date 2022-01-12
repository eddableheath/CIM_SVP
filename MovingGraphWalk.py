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


class WalkExperiment:

    """
        Class for the walk experiment.
    """

    def __init__(self, lattice_basis, prop_density, graph_coords, graph_bounds):
        self.basis = lattice_basis
        self.dimension = self.basis.shape[0]
        self.int_lattice_bounds = self.dimension * np.log2(self.dimension) + np.log2(np.linalg.det(self.basis))

        self.prob_density = prop_density
        self.graph_coords = graph_coords
        self.graph_bounds = graph_bounds

        self.current_integer_vector = np.zeros(self.dimension)

    def compute_overspill(self):
        overspill_amounts = np.zeros_like(self.current_integer_vector)
        for i in range(self.current_integer_vector.shape[0]):
            if abs(self.current_integer_vector[i]) + self.graph_bounds > self.int_lattice_bounds:
                overspill_amounts[i] = np.sign(self.current_integer_vector[i]) * (abs(self.current_integer_vector[i]) +
                                                                                  self.graph_bounds -
                                                                                  self.int_lattice_bounds)
        return overspill_amounts

    def compute_alt_prob_density(self, overspill):
        grid_graph = nx.grid_graph(dim=[2*self.graph_bounds + 1 - overspill[i]
                                        for i in range(self.dimension)])


    def update_state(self):
        overspill = self.compute_overspill()
        if sum(overspill) == 0:
            dist = self.prob_density
            coords = self.graph_coords
        else:
            x=1
        self.current_integer_vector = self.current_integer_vector + np.asarray(coords[np.random.choice(dist.shape[0],
                                                                                                       p=dist)])