# Experiment Classes
# Author: Edmund Dable-Heath
"""
    Experiment classes for the CIM
"""

import numpy as np
import pandas as pd
import isingify
from sim_funcs import simulation_step, normalisation_factor, ising_energy
import construct_basis as cb
from copy import copy


class CIM:

    """
        Main class for CIM experiments
    """

    def __init__(self, **pars):
        """
            Parameter list:
                - control_sys: Error control system for CIM, str
                - couplings: couplings matrix, float-(m, m)-ndarray
                - pump_start: pump parameter start, float
                - pump_end: final pump param value, float
                - tamp_start: initial tamp value, float
                - tamp_end: final tamp value, float
                - beta: beta parameter, float
                - c: (optional for SFC) c parameter, float
                - k: (optional for SFC) k parameter, float
                - limit_func: choice of limiting function, pyfunc
                - nl_func: (optional for SFC) non-linear control function for SFC, pyfunc
                - time_diff: time step multiplier, float
                - iters: max iterations per run, int
                - var_iters: iterations for varying varied-parameters over, int
                - stat_iters: iterations to keep varied-parameters stationary, int
                - repeats: number of repeats of experiment, int
                - seed: randomness seed, int
        """
        # RNG
        self.rng = np.random.default_rng(pars['seed'])

        # Control system
        self.control_sys = pars['control_sys']

        # Couplings
        self.couplings = pars['couplings']
        self.norm_factor = normalisation_factor(self.couplings)
        self.problem_size = self.couplings.shape[0]

        # Varying system params
        self.pump = pars['pump_start']
        self.pump_start = pars['pump_start']
        self.pump_end = pars['pump_end']
        self.tamp = pars['tamp_start']
        self.tamp_start = pars['tamp_start']
        self.tamp_end = pars['tamp_end']

        # Fixed system params
        self.beta = pars['beta']
        if self.control_sys == 'SFC':
            self.c = pars['c']
            self.k = pars['k']

        # Chosen input functions
        self.limit_func = pars['limit_func']
        if self.control_sys == 'SFC':
            self.nl_func = pars['nl_func']

        # Time step params
        self.timediff = pars['time_diff']
        self.iters = pars['iters']
        self.var_iters = pars['var_iters']
        self.stat_iters = pars['stat_iters']
        self.current_it = 0
        self.pump_sched = [self.pump_start + (abs(self.pump_end-self.pump_start)/self.var_iters)*i
                           for i in range(self.var_iters)] + [self.pump_end for _ in range(self.stat_iters)]
        self.tamp_sched = [self.tamp_start + (abs(self.tamp_end-self.tamp_start)/self.var_iters)*i
                           for i in range(self.var_iters)] + [self.tamp_end for _ in range(self.stat_iters)]

        # Experiment params
        self.repeats = pars['repeats']
        self.current_repeat = 0
        self.spins = np.zeros(self.couplings.shape[0])
        self.error = np.zeros(self.couplings.shape[0])
        self.results = np.zeros((self.repeats, self.problem_size))
        self.save_traj = pars['save_traj']
        if self.save_traj:
            self.traj_spins = np.zeros((self.repeats, self.iters, self.problem_size))
            self.traj_error = np.zeros_like(self.traj_spins)

    def reset_exp(self):
        """
            Reset the spins and varying parameters for a new experiment run
        """
        self.pump = self.pump_start
        self.tamp = self.tamp_start
        self.spins = self.rng.normal(0.0, 0.5, self.problem_size)
        self.error = np.ones(self.problem_size)
        self.current_it = 0

    def update_state(self):
        """
            Update spins and error based on simulation step.
        """
        pars = {
            'error_control': self.control_sys,
            'couplings': self.couplings,
            'normalisation': self.norm_factor,
            'spins': self.spins,
            'error': self.error,
            'tamp': self.tamp_sched[self.current_it],
            'pump': self.pump_sched[self.current_it],
            'beta': self.beta,
            'time_diff': self.timediff,
            'limit_func': self.limit_func
             }
        if self.control_sys == 'SFC':
            pars['k'] = self.k
            pars['c'] = self.c
            pars['nl_func'] = self.nl_func
        self.spins, self.error = simulation_step(**pars)

    def run(self):
        for i in range(self.repeats):
            self.reset_exp()
            for j in range(self.iters):
                print(f'at iteration {self.current_it} the spins are {self.spins} and error is {self.error}')
                self.update_state()
                if self.save_traj:
                    self.traj_spins[self.current_repeat, self.current_it] = self.spins
                    self.traj_error[self.current_repeat, self.current_it] = self.error
                    print(self.traj_spins)
                self.current_it += 1
            self.results[self.current_repeat] = self.spins
            self.current_repeat += 1
        return self.results


class CIM_SVP(CIM):

    """
        CIM SVP solver class
    """

    def __init__(self, **pars):
        """
            Parameter list:
                - basis: lattice basis for defining problem, int-(m, m)-ndarray
                - sitek: integer range to search over, int
                - qudit_mapping: qudit encoding of integers by qubits, str
                - requires all other parameters included in standard CIM class other than couplings_matrix.
        """
        # Lattice params
        self.basis = pars['basis']
        self.dim = self.basis.shape[0]
        self.gramm = self.basis @ self.basis.T
        self.sitek = pars['sitek']
        self.qudit_mapping = pars['qudit_mapping']
        self.couplings, self.identity_coeff, self.qubits_per_qudit, self.mu = None, None, None, None
        self.preprocess()

        # Passing to super init
        pars['couplings'] = self.couplings

        CIM.__init__(self, **pars)

    def preprocess(self):
        """
            Compute couplings matrix, number of qubits required per encoding as well as identity and mu coefficients.
        """
        if self.qudit_mapping == 'poly':
            self.couplings, self.identity_coeff = isingify.poly_couplings(self.gramm, self.sitek)
            self.qubits_per_qudit = int(np.ceil(
                (np.sqrt(16 * self.sitek + 1) - 1) / 2
            ))
            self.mu = np.ceil(self.qubits_per_qudit / 2) % 2
        elif self.qudit_mapping == 'bin':
            self.couplings, self.identity_coeff = isingify.bin_couplings(self.gramm, self.sitek)
            self.qubits_per_qudit = int(np.ceil(np.log2(self.sitek)) + 1)
            self.mu = 1
        elif self.qudit_mapping == 'ham':
            self.couplings, self.identity_coeff = isingify.ham_couplings(self.gramm, self.sitek)
            self.qubits_per_qudit = 2 * self.sitek
            self.mu = 0

    def bitstr_to_coeff_vector(self, bitstr):
        """
            Convert a list of spins (+/- 1) interpreted as a bitstring to a lattice coefficient vector. Note the poly
            encoding does not require changing to the {0, 1} basis.
        :param bitstr: list of spin states interpreted as bitstring through operator (1-s)/2, (m, )-ndarray
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
            return vect - 2**(k-1)

        elif self.qudit_mapping == 'ham':
            k = self.qubits_per_qudit
            vect = np.zeros(self.dim)
            psi = (1 - bitstr) / 2
            for i in range(self.dim):
                vect[i] = np.sum(psi[i*k:(i+1)*k])
            return np.array(vect) - np.ceil(k/2)

        elif self.qudit_mapping == 'poly':
            k = self.qubits_per_qudit
            mu = int(np.ceil(k/2)) % 2
            vect = np.zeros(self.dim)
            for i in range(self.dim):
                vect[i] = int(np.sum(
                    bitstr[i*k: (i+1)*k] * (np.arange(1, k+1)/2)
                ) + (mu/2))
            return vect

    def set_last_spin_1(self):
        """
            If an auxillary spin is required we should keep it set to 1.
        """
        if self.mu:
            # self.spins[-1] = self.limit_func(self.tamp_sched[self.current_it])
            self.spins[-1] = abs(self.spins[-1])

    def postprocess(self):
        """
            Translate back to lattice problem.
        """
        int_vects = [self.bitstr_to_coeff_vector(np.sign(spins).astype(int))
                     for spins in self.results]
        latt_vects = [np.dot(self.basis.T, vect) for vect in int_vects]
        norms = np.around(np.linalg.norm(latt_vects, axis=1), 3)
        ising_energies = [ising_energy(np.sign(spins), self.couplings) + self.identity_coeff
                          for spins in self.results]
        print(self.traj_spins)
        print(self.traj_spins.shape)
        if self.save_traj:
            print(self.results)
            print(self.spins)
            print()
            print(self.traj_spins[0][-1])
            print(ising_energy(np.sign(self.traj_spins[0][-1]), self.couplings))
            traj_energies = [ising_energy(np.sign(self.traj_spins[0][i]), self.couplings) + self.identity_coeff
                             for i in range(self.traj_spins[0].shape[0])]
            print(traj_energies)
            print(len(traj_energies))
            df = pd.DataFrame(data=list(zip(
                np.around(self.results, 2),
                int_vects,
                latt_vects,
                norms,
                ising_energies,
                [traj_energies]
            )),
                columns=[
                    'spins',
                    'integer_vectors',
                    'lattice_vectors',
                    'norms',
                    'ising_energies',
                    'traj_energies'
                ])
        else:
            df = pd.DataFrame(data=list(zip(
                np.around(self.results, 2),
                int_vects,
                latt_vects,
                norms,
                ising_energies
            )),
                columns=[
                    'spins',
                    'integer_vectors',
                    'lattice_vectors',
                    'norms',
                    'ising_energies'
            ])
        return df

    def run(self, results_type='return', filename='test.csv'):
        """
            Overriding general CIM run method to include lattice specific issues
        """
        for i in range(self.repeats):
            self.reset_exp()
            self.set_last_spin_1()
            for j in range(self.iters):
                # print(f'at iteration {self.current_it} the spins are {self.spins} and error is {self.error}')
                self.update_state()
                self.set_last_spin_1()
                if self.save_traj:
                    self.traj_spins[self.current_repeat, self.current_it] = self.spins
                    self.traj_error[self.current_repeat, self.current_it] = self.error
                    # print(self.traj_spins)
                self.current_it += 1
            self.results[self.current_repeat] = self.spins
            self.current_repeat += 1

        results = self.postprocess()
        path = '.'

        if results_type == 'return':
            return results
        elif results_type == 'write':
            results.to_csv(f'{path}/{filename}')


class CIM_SVP_basis_update(CIM_SVP):

    """
        CIM SVP solver with basis updating.
    """

    # Todo: compute sitek based on orthogonality defect bound from albrecht?

    def __init__(self, **pars):
        CIM_SVP.__init__(self, **pars)
        if pars['basis_comparison'] == 'mean':
            self.basis_comp = cb.mean_comp
        elif pars['basis_comparison'] == 'median':
            self.basis_comp = cb.med_comp
        elif pars['basis_comparison'] == 'max':
            self.basis_comp = cb.max_comp
        self.bases = []
        self.bases.append(copy(self.basis))
        self.search_buffer = 0
        self.search_buffer_window = pars['search_buffer'] * pars['iters']

    def update_basis(self, new_vector):
        """
            If shorter vector than basis vectors is found then update basis and couplings matrix.
        """
        prim_vec = cb.check_for_primitivity(new_vector)
        self.basis = cb.construct_basis(prim_vec, self.basis)
        self.gramm = self.basis @ self.basis.T
        self.preprocess()  # updates couplings matrix etc.

    def run(self, results_type='return', filename='test.csv'):
        """
            Overrides run from standard CIM SVP solver to include basis updater routine.
        """
        for i in range(self.repeats):
            print(f'resetting experiment, current basis:')
            print(self.basis)
            self.reset_exp()
            self.basis = copy(self.bases[0])
            print('reset basis:')
            print(self.basis)
            self.bases.append(copy(self.basis))
            self.set_last_spin_1()
            for j in range(self.iters):
                self.update_state()
                self.set_last_spin_1()
                if self.save_traj:
                    self.traj_spins[self.current_repeat, self.current_it] = self.spins
                    self.traj_error[self.current_repeat, self.current_it] = self.error
                self.current_it += 1
                if self.search_buffer > self.search_buffer_window:
                    int_vec = self.bitstr_to_coeff_vector(np.sign(self.spins)).astype(int)
                    latt_vec = np.dot(self.basis.T, int_vec).astype(int)
                    if self.basis_comp(self.basis, latt_vec) and np.linalg.norm(latt_vec) > 0:
                        print(f'new basis vector to be added: {latt_vec} ---------------------------------')
                        old_basis_det = np.linalg.det(self.basis)
                        mean_basis_length = np.mean(np.linalg.norm(self.basis, axis=1))
                        self.update_basis(int_vec)
                        self.bases.append(copy(self.basis))
                        self.search_buffer = 0
                        print(f'length of new vector {np.linalg.norm(latt_vec)}')
                        print(f'checking bases construction worked')
                        print(f'original basis det: {np.linalg.det(self.bases[0])}')
                        print(f'old basis det: {old_basis_det}')
                        print(f'new basis det: {np.linalg.det(self.basis)}')
                        print(f'old basis mean length: {mean_basis_length}')
                        print(f'new basis mean length: {np.mean(np.linalg.norm(self.basis, axis=1))}')
                        print(f'new basis:')
                        print(self.basis)
                        print(f'new couplings:')
                        print(self.couplings)
                        print(f'new identity coeff')
                        print(self.identity_coeff)
                    else:
                        self.search_buffer += 1
                else:
                    self.search_buffer += 1
            self.results[self.current_repeat] = self.spins
            self.current_repeat += 1

        results = self.postprocess()
        path = '.'

        if results_type == 'return':
            return results, self.bases
        elif results_type == 'write':
            results.to_csv(f'{path}/{filename}')





