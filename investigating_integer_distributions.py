# Investigating pre lattice integer distribution
# Author: Edmund Dable-Heath
# When traversing the lattice space the clear way to actually get around is via the integer coefficients. So what does
# the space of integers look like wrt the lattice? The map (z_1, z_2) --> (l_1, l_2); (l_1, l_2) = B(z_1, z_2) for basis
# matrix B should reveal the distribution of norms for integer points. This is the space/distribution that needs to be
# explored, or at least would be most easily explored by an updating procedure such as DHMC, so what does this
# distribution look like?

import numpy as np
from matplotlib import pyplot as plt
import math
from mpl_toolkits.mplot3d.axes3d import Axes3D


def integer_to_lattice_plotter(basis, shortest_vector):
    def range_calc(basis):
        dimension = basis.shape[0]
        return dimension * math.log2(dimension) + math.log2(abs(np.linalg.det(basis)))
    int_range = int(range_calc(basis))
    int_x, int_y = np.mgrid[-int_range:int_range, -int_range: int_range]
    z = np.zeros((2*int_range, 2*int_range))
    for i in range(int_x.shape[0]):
        for j in range(int_x.shape[0]):
            if np.linalg.norm(int_x[i][j]*basis[0] + int_y[i][j]*basis[1]) > 0:
                z[i][j] = np.linalg.norm(int_x[i][j]*basis[0] + int_y[i][j]*basis[1])
            else:
                z[i][j] = 0

    # shortest vector algebraic solution
    shortest_ints = np.linalg.solve(basis.T, shortest_vector)

    # Plotting
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(int_x, int_y, z, cmap='viridis')
    ax.plot3D([-shortest_ints[0], -shortest_ints[0]],
              [-shortest_ints[1], -shortest_ints[1]],
              [0, np.max(z)], 'gray')
    ax.scatter(-shortest_ints[0], -shortest_ints[1], np.max(z))
    ax.plot3D([shortest_ints[0], shortest_ints[0]],
              [shortest_ints[1], shortest_ints[1]],
              [0, np.max(z)], 'gray')
    ax.scatter(shortest_ints[0], shortest_ints[1], np.max(z))
    plt.show()


if __name__=="__main__":
    latt_hnf = np.array([[32, 0], [-6, 1]])
    latt_lll = np.array([[2, 5], [-6, 1]])
    sv = np.array([2, 5])
    integer_to_lattice_plotter(latt_hnf, sv)
    integer_to_lattice_plotter(latt_lll, sv)