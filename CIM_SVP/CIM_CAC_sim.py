# CIM CAC simulations functions
# Author: Edmund Dable-Heath
# Here is all of the functions needed for the actual simulation, an extra module to keep it all organised in one place.


import numpy as np
import matplotlib.pyplot as plt
from copy import copy


def compute_ground_E(interactions):
    """
        Attempt to compute the ground energy state based on the interaction matrix.
    :param interactions: interactions matrix, n x n numpy array
    :return: ground state energy?
    """
    problem_size = interactions.shape[0]
    sol_base_vectors = np.zeros([problem_size**2, problem_size])

    for idx in range(problem_size**2):
        str_bin = format(idx, "0"+str(problem_size)+"b")  # make binary string
        for jdx in range(problem_size):
            sol_base_vectors[idx][jdx] = 1 - 2*int(str_bin[jdx])

    # initialise values
    min_energy = problem_size**2
    upper_bound = problem_size**2

    for sol in sol_base_vectors:
        energy = np.dot(np.dot(interactions, sol), sol)
        if energy <= min_energy + 0.000000001:
            min_energy = energy
        elif energy < upper_bound:
            upper_bound = energy

    return min_energy


def std_energy(interactions, spins, pumpfield, normalisation):
    """
        Energy computation at specific step
    :param interactions: J_ij, square numpy array
    :param spins: spin variable, numpy array
    :param pumpfield: pumpfield parameter, float
    :param normalisation: normalisation of Ising summand
    :return: energy
    """
    return np.sum((spins*spins)/4.0 - (pumpfield*spins*spins)/2.0 + normalisation*spins*np.dot(interactions, spins))


def E_ising(interactions, spins, identity_coeff):
    """
        Ising energy computation
    :param interactions: J_ij, square numpy array
    :param spins: spin variable, numpy array
    :return: Ising energy
    """
    return np.sum(np.sign(spins)*np.dot(interactions, np.sign(spins))) + identity_coeff


def limit_range(spin, tamp):
    """
        # todo: workout what the function does
    :param spin: spin variable for state of ising model, np array
    :param tamp: tamp, float
    :return: ndarray, but of what?
    """
    return np.minimum(np.maximum(spin, -np.sqrt(tamp)*1.5), np.sqrt(tamp)*1.5)


def simulation_step(interaction_terms, spin, error, normalisation, beta, pumpfield, timestep, t_amp):
    """
        Main step in the simulation, for a given timestep simulates the system given by control equations.
    :param interaction_terms: the J_ij matrix for the ising problem, square numpy array
    :param spin: spin variable giving value for spins at each site, numpy array
    :param error: error pulse amplitude, giving the error in the system, numpy array
    :param normalisation: normalisation coefficient for Ising summand, float
    :param beta: general parameter, float
    # todo: either check with Sam or work out from paper
    :param timestep: time step for numerical integration, float
    :param pumpfield: p-1 from paper, general parameter of the system, float
    :param t_amp: t amplitude
    # Todo: check with Sam what this is again
    :return: integration step, a tuple of the next evolution in both cases.
    """
    ising_summation = normalisation * np.dot(interaction_terms, spin) * error
    spin_diff = -spin**3 + pumpfield * spin - ising_summation

    error_diff = -beta * error * (spin**2 - t_amp)

    return limit_range(spin + spin_diff*timestep, t_amp), error + error_diff*timestep


def trajectories(interaction_terms, spins, error, normalisation, beta, timestep, timesplit,
                 pumpfield_start, pumpfield_raise, t_amp_start, t_amp_raise, max_viable_energy=None):
    """
        Simulating a trajectory through the relevant solution space.
    :param interaction_terms: J_ij, square numpy array
    :param spins: spin variable giving value for spin at each site, numpy array
    :param error: error pulse amplitude, numpy array
    :param normalisation: normalisation coeff for Ising summand
    :param beta: general parameter, float
    :param timestep: time step for step of simulation, float
    :param timesplit: number of time steps to run through, int
    :param pumpfield_start: starting pumpfield parameter, p-1, float
    :param pumpfield_raise: range of pumpfield parameter over sim, float
    :param t_amp_start: starting t amplitude, float
    :param t_amp_raise: range of t_amp over sim, float
    :param max_viable_energy: start outputting the spins when the energy gets below the max viable energy
    :return: tuple of updated spins, error and energy
    """
    E_opt = 0.
    small_energies = []
    zeros_count = 0
    print(f'initial energy: {E_ising(interaction_terms, spins)}')
    for i in range(timesplit):
        pumpfield = pumpfield_start + (float(i) / timesplit) * pumpfield_raise
        t_amp = t_amp_start + (float(i) / timesplit) * t_amp_raise
        spins, error = simulation_step(interaction_terms, spins, error, normalisation, beta, pumpfield, timestep, t_amp)
        E_opt = min(E_ising(interaction_terms, spins), E_opt)
        if max_viable_energy is not None:
            if 0. < E_ising(interaction_terms, spins) < max_viable_energy:
                small_energies.append((E_ising(interaction_terms, spins), spins))
            elif E_ising(interaction_terms, spins) == 0.:
                zeros_count += 1.
    print(f'small energies {small_energies}')
    print(f'zeros count {zeros_count}')

    return spins, error, E_opt


def plot_trajectories(interaction_terms, spins, error, normalisation, beta, timestep, timesplit, pumpfield_start,
                      pumpfield_raise, t_amp_start, t_amp_raise, ground_E, min_found_energy):
    """
        plotting the trajectories through the solution spaces
    :param interaction_terms: J_ij, square numpy array
    :param spins: spin variable giving value for spin at each site, numpy array
    :param error: error pulse amplitude, numpy array
    :param normalisation: normalisation coeff for Ising summand
    :param beta: general parameter, float
    :param timestep: time step for step of simulation, float
    :param timesplit: number of time steps to run through, int
    :param pumpfield_start: starting pumpfield parameter, p-1, float
    :param pumpfield_raise: range of pumpfield parameter over sim, float
    :param t_amp_start: starting t amplitude, float
    :param t_amp_raise: range of t_amp over sim, float
    :param ground_E: pre computed reference ground energy, float
    :param min_found_energy: minimum energy found from simulation, float
    :return: plot of the trajectories
    """
    problem_size = interaction_terms.shape[0]
    spin_results = np.zeros((problem_size, timesplit))
    error_results = np.zeros((problem_size, timesplit))
    ising_energy_results = np.zeros(timesplit)
    for i in range(timesplit):
        pumpfield = pumpfield_start + (float(i)/timesplit) * pumpfield_raise
        t_amp = t_amp_start + (float(i)/timesplit) * t_amp_raise
        spins, error, = simulation_step(interaction_terms, spins, error, normalisation,
                                        beta, pumpfield, timestep, t_amp)
        spin_results[:, i] = spins
        error_results[:, i] = error
        ising_energy_results[i] = E_ising(interaction_terms, spins)

    plt.title("Spin Trajectory")
    x_axis = np.array(range(timesplit)) * timestep
    plt.ylim((-2.5, 2.5))
    for i in range(problem_size):
        plt.plot(x_axis, spin_results[i, :])
    plt.show()
    plt.close()

    plt.title("Error Trajectory")
    x_axis = np.array(range(timesplit)) * timestep
    plt.ylim((0, 5.0))
    for i in range(problem_size):
        plt.plot(x_axis, error_results[i, :])
    plt.show()
    plt.close()

    plt.title("Residual Ising Energy Trajectory")
    x_axis = np.array(range(timesplit))*timestep
    plt.ylim((0, 10.0))
    E_ref = copy(ground_E)
    if E_ref > 0:
        E_ref = min_found_energy
    print(np.min(ising_energy_results))
    plt.plot(x_axis, ising_energy_results - E_ref)
    plt.show()


def rounding_spins(spin):
    """
        rounding the spins to +/- 1, single spin version
    :param spin: a single spin value, float
    :return: +/- dependin on how close it is
    """
    if spin >= 0:
        return 1
    else:
        return -1


rounding_spins_vectorised = np.vectorize(rounding_spins)

# Testing functions
# if __name__ == "__main__":
