# Experiemnt classes
# Author: Edmund Dable-Heath
#
# Here is the main class used to instantiate the experiment.

import numpy as np
import pandas as pd

import isingify as lf
import CIM_CAC_sim as sim
from copy import copy
import matplotlib.pyplot as plt


class CIM_CAC_experiment:

    """
        Main class for the CIM CAC experiment.
    """

    def __init__(self, interaction_terms, t_amp_start, t_amp_raise, beta, pumpfield_start, pumpfield_raise, time_step,
                 time_splits, no_repetitions, results_tolerance, ground_energy=None):
        """
        :param interaction_terms: interactions matrix (n, n) numpy array encoding (ZZ) interactions of ising model
        :param t_amp_start: t amplitude starting value, float
        :param t_amp_raise: t amplitude raise value, float
        :param beta: #todo: clarify with Sam what this parameter does
        :param pumpfield_start: pumpfield starting value, float
        :param pumpfield_raise: pumpfield raised value, float
        :param time_step: time step between integration steps, float
        :param time_splits: total number of time steps in simulation, int
        :param no_repetitions: number of repetitions of experiment, int
        :param results_tolerance: results tolerance, float
        :param ground_energy: pre computed ground energy, float
        """
        self.J_ij = interaction_terms
        self.problem_size = self.J_ij.shape[0]
        self.t_amp_start = t_amp_start
        self.t_amp_raise = t_amp_raise
        self.beta = beta
        self.pumpfield_start = pumpfield_start
        self.pumpfield_raise = pumpfield_raise
        self.pumpfield_end = pumpfield_start + pumpfield_raise
        self.time_step = time_step
        self.time_splits = time_splits
        self.repetitions = no_repetitions
        self.tolerance = results_tolerance

        # instantiate in memory the spins and errors, to be generate fully when needed.
        self.spins = np.zeros(self.problem_size)
        self.error = np.zeros(self.problem_size)

        # optionally compute ground energy for small problems
        if ground_energy is None:
            if self.problem_size < 16:
                self.ground_E = sim.compute_ground_E(self.J_ij)
            else:
                print("Incorrect ground energy value")
        else:
            self.ground_E = ground_energy

        # Compute normalisation coefficient
        Jij_sum = 0.
        for i in range(self.problem_size):
            for j in range(i):
                Jij_sum += abs(self.J_ij[i, j])
        self.norm_coeff = 1. / np.sqrt(Jij_sum / self.problem_size)

    def trajectories(self, save_trajectory=False):
        """
            Simulating a trajectory through the relevant solution space.
        :param save_trajectory: (default=False) save the trajectories of the simulation not just the final state.
        :return:
                final state (spins, errors, e_opt) of simulation over all time steps if save_trajectory=False
        """
        E_opt = 0.
        traj = []
        for i in range(self.time_splits):
            pumpfield = self.pumpfield_start + (float(i) / self.time_splits) * self.pumpfield_raise
            t_amp = self.t_amp_start + (float(i) / self.time_splits) * self.t_amp_raise
            self.spins, self.error = sim.simulation_step(self.J_ij, self.spins, self.error, self.norm_coeff, self.beta,
                                                         pumpfield, self.time_step, t_amp)
            E_opt = min(sim.E_ising(self.J_ij, self.spins), E_opt)
            if save_trajectory:
                traj.append((copy(self.spins), copy(self.error), E_opt))

        if save_trajectory:
            return traj
        else:
            return copy(self.spins), copy(self.error), E_opt

    def plot_trajectories(self):
        """
            plotting the trajectories through the spin space and error space.
        """
        spin_results = np.zeros((self.problem_size, self.time_splits))
        error_results = np.zeros((self.problem_size, self.time_splits))
        ising_energy_results = np.zeros(self.time_splits)
        simulation_results = self.trajectories(save_trajectory=True)
        for i in range(len(simulation_results)):
            spin_results[:, i] = simulation_results[i][0]
            error_results[:, i] = simulation_results[i][1]
            ising_energy_results = sim.E_ising(self.J_ij, simulation_results[i][0])

        min_found_energy = np.min([result[2] for result in simulation_results])

        # TODO: print statements to be replaced with writing results to file
        print(f'print some for the results for the trajectories as well')

        plt.title("Spin Trajectory")
        x_axis = np.array(range(self.time_splits)) * self.time_step
        plt.ylim((-2.5, 2.5))
        for i in range(self.problem_size):
            plt.plot(x_axis, spin_results[i, :])
        plt.show()
        plt.close()

        plt.title("Error Trajectory")
        x_axis = np.array(range(self.time_splits)) * self.time_step
        plt.ylim((0, 5.0))
        for i in range(self.problem_size):
            plt.plot(x_axis, error_results[i, :])
        plt.show()
        plt.close()

        plt.title("Residual Ising Energy Trajectory")
        x_axis = np.array(range(self.time_splits)) * self.time_step
        plt.ylim((0, 10.0))
        E_ref = copy(self.ground_E)
        if E_ref > 0:
            E_ref = min_found_energy
        # print(np.min(ising_energy_results))
        plt.plot(x_axis, ising_energy_results - E_ref)
        plt.show()

    def run(self, plots=False):
        """
            run the simulation, with an option for plotting the trajectories as well
        :return: currently a print statement.
        """
        if plots:
            self.spins = np.random.rand(self.problem_size) - 0.5
            self.error = 10 * np.ones(self.problem_size)
            self.plot_trajectories()

        # TODO: update from print statements to writing to file
        else:
            sucesses = 0
            sucess_results = []
            for i in range(self.repetitions):
                print("------------------------")
                self.spins = np.random.rand(self.problem_size) - 0.5
                print(f"initial spins: {self.spins}")
                self.error = 10*np.ones(self.problem_size)
                self.spins, _, E_opt = self.trajectories()
                print(f"optimal energy: {E_opt - self.ground_E}")
                print(f"energy: {sim.E_ising(self.J_ij, self.spins)}")
                print(f"final spins: {self.spins}")
                sucess_results.append(E_opt)

                if abs(E_opt - self.ground_E) < self.tolerance:
                    sucesses += 1

            min_found = np.min(sucess_results)
            print("-----------------------------")
            print(f"sucess rate: {float(sucesses) / self.repetitions}")
            print()
            print(f"min found: {min_found}")
            print(f"frequency: {float(np.sum(np.array(sucess_results) == min_found) / self.repetitions)}")


class CIM_CAC_SVP_experiment(CIM_CAC_experiment):

    def __init__(self, lattice_basis, sitek, qudit_mapping, t_amp_start, t_amp_raise, beta, pumpfield_start,
                 pumpfield_raise, time_step, time_splits, no_repetitions, results_tolerance, ground_energy=None):
        """
            init function for the class
        :param lattice_basis: lattice basis, (m, m) numpy array encoding entire lattice problem
        :param sitek: range of integer encoding, int
        :param qudit_mapping: choice of qudit encoding, string
        """
        self.basis = lattice_basis
        self.dim = self.basis.shape[0]  # dimension of lattice problem
        self.gramm = lattice_basis@lattice_basis.T  # The gramm matrix
        self.k = sitek
        self.qudit_mapping = qudit_mapping
        if self.qudit_mapping == 'bin':
            self.qubits_per_qudit = int(np.ceil(np.log2(2*self.k)))
        elif self.qudit_mapping == 'poly':
            self.qubits_per_qudit = int(np.ceil(
                (np.sqrt(16 * sitek + 1) - 1) / 2
            ))
        elif self.qudit_mapping == 'ham':
            self.qubits_per_qudit = 2 * self.k

        self.spin_results = []
        self.J_ij, self.identity_coeff = self.preprocess()
        CIM_CAC_experiment.__init__(self, self.J_ij, t_amp_start, t_amp_raise, beta, pumpfield_start,
                                    pumpfield_raise, time_step, time_splits, no_repetitions, results_tolerance,
                                    ground_energy)

    def bitstr_to_coeff_vector(self, bitstr):
        """
            Convert a list of spins (+/- 1) interpreted as a bitstring to a lattice coefficient vector
        :param bitstr: list of spin states interpreted as bitstring through operator (1-s)/2
        :return: comma-delimited coefficient vector as a string
        """

        if self.qudit_mapping == 'bin':
            k = self.qubits_per_qudit
            vect = np.zeros(self.dim)
            psi = (1 - bitstr)/2
            for i in range(self.dim):
                vect[i] = int(np.sum(
                    psi[i*k:(i+1)*k] * 2**np.arange(k)
                ))
            return vect

        elif self.qudit_mapping == 'ham':
            ind = 0
            vect = []
            for i in range(self.dim):
                num = 0
                for j in range(int(self.qubits_per_qudit)):
                    num += bitstr[ind] / 2
                    ind += 1
                vect.append(num)
            return np.array(vect)

        elif self.qudit_mapping == 'poly':
            k = self.qubits_per_qudit
            mu = int(np.ceil(k/2)) % 2
            vect = np.zeros(self.dim)
            for i in range(self.dim):
                vect[i] = int(np.sum(
                    bitstr[i*k: (i+1)*k] * (np.arange(1, k+1)/2)
                ) + (mu/2))
            return vect

    def preprocess(self):
        """
            Take the lattice properties and map them to relevant properties used by the CIM, specifically the
            interaction matrix J_ij
        """

        if self.qudit_mapping == "bin":
            return lf.bin_couplings(self.gramm, self.k)
        elif self.qudit_mapping == 'ham':
            return lf.ham_couplings(self.gramm, self.k)
        elif self.qudit_mapping == 'poly':
            return lf.poly_couplings(self.gramm, self.k)

    def execute(self):
        """
            run the simulation of the CIM SVP algorithm
        """
        for i in range(self.repetitions):
            self.spin_results.append(self.trajectories()[0])

    def postprocess(self):
        """
            post process the results from CIM to SVP
        """
        print(self.spin_results[0])
        print(sim.rounding_spins_vectorised(self.spin_results[0]))
        print(sim.E_ising(self.J_ij, sim.rounding_spins_vectorised(self.spin_results[0])))
        print(self.identity_coeff)
        print(sim.E_ising(self.J_ij, sim.rounding_spins_vectorised(self.spin_results[0])) + self.identity_coeff)
        int_vects = [self.bitstr_to_coeff_vector(sim.rounding_spins_vectorised(spins).astype(int))
                     for spins in self.spin_results]
        latt_vects = [np.dot(self.basis.T, vect) for vect in int_vects]
        norms = [np.linalg.norm(vect) for vect in latt_vects]
        ising_energies = [sim.E_ising(self.J_ij, sim.rounding_spins_vectorised(spins)) + self.identity_coeff
                          for spins in self.spin_results]
        df = pd.DataFrame(data=list(zip(np.around(self.spin_results, 2), int_vects, latt_vects, norms, ising_energies)),
                          columns=['spins', 'integer vectors', 'lattice vectors', 'norms', 'ising energies'])
        return df

    def run(self, result_type='return', plots=False):
        """
            run the CIM SVP experiment: run and post process

        :param result_type: (default='return') choose to return the data frame containing the results or to just write
                            the results to a results folder
        :param plots: (default=False) also plot the spin trajectories
        """
        self.spins = np.random.randn(self.problem_size) - 0.5
        self.spins[-1] = 1
        self.error = 10*np.ones(self.problem_size)
        self.execute()
        results = self.postprocess()
        path = '.'

        if plots:
            self.plot_trajectories()

        if result_type == 'return':
            return results
        elif result_type == 'write':
            results.to_csv(f'{path}/test.csv')
