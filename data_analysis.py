# Analysing the Data
# Author: Edmund Dable-Heath
# Analysing the data from the quantum walk experiments. The goal here is to find how fast the algorithm has mixed. There
# are three qualities to analyse in the data:
#
# 1. How correlated with the start point the chain is.
#
# 2. A goodness of fit test to the target distribution for a lag of N points in the data.
#
# 3. A snapshot over the multiple runs gives a sample distribution at a given point from the distribution, examining the
#    mean squared error between each step should give a good idea of the speed of convergence.
#
# This should all be compared against the different propagation times on the same graph.
#
# Currently this is being done from lattice 0, dimension 2 but this will be


import numpy as np
import scipy as sp
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt


def mean_squared_errors(times, path):
    """
        Gives a dictionary with the mean squared error from slice to slice across all 100 runs for each stated time
        parameter.
    :param times: time parameters tobe examined
    :param path: path to results organised by time, looks like 'initial_results/lattice_#/lattice_type/'
    :return: dictionary {time_param : [list of mean_squared_errors]}
    """
    results = {}
    for time in times:
        new_path = path+'t='+str(time)+'/'
        initial_vectors = [np.genfromtxt(new_path+'/'+str(file), delimiter=',')
                           for file in os.listdir(new_path)]
        slices_array = [np.array([array[index]
                                  for array in initial_vectors])
                        for index in range(len(initial_vectors[0]))]
        results[time] = [mean_squared_error(slices_array[i-1], slices_array[i])
                         for i in range(1, len(slices_array))]
    return results


def clean_up(times, path):
    for time in times:
        new_path = path+'t='+str(time)+'/'
        results = []
        dir_list = os.listdir(new_path)
        results.append(np.genfromtxt(new_path+str(dir_list[0]), delimiter=','))
        dir_list.pop(0)
        for file in dir_list:
            new_result = np.genfromtxt(new_path+str(file), delimiter=',')
            checksum = []
            for result in results:
                if (new_result == result).all():
                    checksum.append(1)
                else:
                    checksum.append(0)
            if sum(checksum) == 0:
                results.append(new_result)
            else:
                os.remove(new_path+file)


def plot_walk(data, time, shortest_vector):
    fig = plt.figure(figsize=(8,8), dpi=200)
    ax = fig.add_subplot(111)

    start = data[:1]
    stop = data[-1:]

    ax.scatter(data[:, 0], data[:, 1], c='blue', alpha=0.25, s=0.05)
    ax.plot(data[:, 0], data[:, 1], c='blue', alpha=0.5, lw=0.25, ls='-')
    ax.plot(start[:, 0], start[:, 1], c='red', marker='+')
    ax.plot(stop[:, 0], stop[:, 1], c='black', marker='o')

    ax.plot(shortest_vector[0], shortest_vector[1], c='yellow', marker='o')
    ax.plot(-shortest_vector[0], -shortest_vector[1], c='yellow', marker='o')

    plt.title('2D CTQW t='+str(time))
    # plt.tight_layout(pad=0)
    plt.show()


if __name__ == "__main__":
    lattice_hnf = np.genfromtxt('Lattices/2/0/hnf.csv', delimiter=',')
    lattice_lll = np.genfromtxt('Lattices/2/0/lll.csv', delimiter=',')
    short_vec = np.genfromtxt('Lattices/2/0/sv.csv')
    time_stamps = np.linspace(0.5, 5, int(5/0.5))
    path = 'initial_results/HNF/'

    walk_data = {}

    for t in time_stamps:
        new_path = path+'t='+str(t)+'/'
        dir_list = os.listdir(new_path)
        new_walk = np.genfromtxt(new_path+str(dir_list[0]), delimiter=',')
        for file in dir_list:
            new_walk = np.append(new_walk, np.genfromtxt(new_path+str(file), delimiter=','), axis=0)
        walk_data[t] = np.dot(new_walk, lattice_hnf)

    mean_results = mean_squared_errors(time_stamps, path)

    # for t in time_stamps:
    #     plot_walk(walk_data[t], t, short_vec)

    for t in time_stamps:
        norms = np.linalg.norm(walk_data[t], axis=1)
        n, c = np.unique(norms, return_counts=True)
        plt.bar(n, c)
        plt.title('Sampled norms for t='+str(t))
        plt.vlines(np.linalg.norm(short_vec), ymin=0, ymax=100000, colors='r')
        for vec in lattice_hnf:
            plt.vlines(np.linalg.norm(vec), ymin=0, ymax=100000, colors='y')
        plt.xlim([0, 50])
        plt.show()

    for t in time_stamps:
        plt.plot(mean_results[t], 'bo')
        plt.title('Mean square error over all runs t='+str(t))
        plt.show()

