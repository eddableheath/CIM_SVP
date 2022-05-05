# Simulation Functions for CIM
# Author: Edmund Dable-Heath
"""
    Collection of functions necessary for each simulation of the CIM. In here are strictly functions for the simulation
    and not for any other admin necessary:
        - normalisation factor
        - Ising energy for given state
        - limit range
        - step of simulation:
            - implement for both CAC and CFC with option to further include other CIMs
"""

import numpy as np

# Global variables
global update_func


def normalisation_factor(couplings):
    """
        Compute the normalisation factor for the couplings matrix.
    :param couplings: coupling matrix, float-(m, m)-ndarray
    :return: normalisation factor, float
    """
    norm = 0.
    for i in range(couplings.shape[0]):
        for j in range(i):
            norm += abs(couplings[i, j])
    return 1. / np.sqrt(norm / couplings.shape[0])


def ising_energy(state, couplings):
    """
        For a given state compute the Ising energy.
    :param state: current spins (use sign to get {-1, 1} state), float-(m, )-ndarray
    :param couplings: couplings matrix, float-(m, m)-ndarray
    :return: Ising energy of current state, float
    """
    return np.dot(np.sign(state), np.dot(couplings, np.sign(state)))


def limit_range(state, tamp, limit_func):
    """
        limit the range of the current spin state
    :param state: current spins, float-(m, m)-ndarray
    :param tamp: current t-amplitude, float
    :param limit_func: choice of limiting function, pyfunc
    :return: bounded spins in state, float-(m, )-ndarray
    """
    bound = limit_func(tamp)
    if bound is None:
        return state
    return np.minimum(np.maximum(state, -bound), bound)


# Spin and error diffs for each error correcting CIM
def CAC_diff(**pars):
    """
       compute the simulation step for the CAC control.
    :param pars: dictionary of parameters for function:
                    - couplings, float-(m, m)-ndarray
                    - spins, float-(m, )-ndarray
                    - error, float-(m, )-ndarray
                    - tamp, float
                    - beta, float
                    - pump, float
                    - normalisation, float
    :return: timestep diff for spins and error terms: float-(m, )-ndarray, float-(m, )-ndarray
    """
    ising_sum = pars['error']*pars['normalisation']*np.dot(pars['couplings'], pars['spins'])
    return (-pars['spins']**3 + pars['pump']*pars['spins'] - ising_sum,
            -pars['beta']*pars['error']*(pars['spins']**2-pars['tamp']))


def CFC_diff(**pars):
    """
       compute the simulation step for the CFC control.
    :param pars: dictionary of parameters for function:
                    - couplings, float-(m, m)-ndarray
                    - spins, float-(m, )-ndarray
                    - error, float-(m, )-ndarray
                    - tamp, float
                    - beta, float
                    - pump, float
                    - normalisation, float
    :return: timestep diff for spins and error terms: float-(m, )-ndarray, float-(m, )-ndarray
    """
    ising_sum = pars['error']*pars['normalisation']*np.dot(pars['couplings'], pars['spins'])
    # print(f'ising sum here: {ising_sum}')
    return (-pars['spins']**3 + pars['pump']*pars['spins'] - ising_sum,
            -pars['beta']*pars['error']*(ising_sum**2-pars['tamp']))


def SFC_diff(**pars):
    """
       compute the simulation step for the SFC control.
    :param pars: dictionary of parameters for function:
                    - couplings, float-(m, m)-ndarray
                    - spins, float-(m, )-ndarray
                    - error, float-(m, )-ndarray
                    - tamp, float
                    - beta, float
                    - pump, float
                    - normalisation, float
                    - k, float
                    - c, float
                    - nl_func, pyfunc
    :return: timestep diff for spins and error terms: float-(m, )-ndarray, float-(m, )-ndarray
    """
    ising_sum = pars['normalisation']*np.dot(pars['couplings'], pars['spins'])
    return (-pars['spins']**3 + pars['pump']*pars['spins'] -
            pars['nl_func'](pars['c']*ising_sum) - pars['k']*(ising_sum-pars['error']),
            -pars['beta']*pars['error']*(ising_sum**2-pars['tamp']))


def simulation_step(**pars):
    """
        single time step of the simulation
    :param pars: dictionary of parameters for function:
                    - error_control, str
                    - couplings, float-(m, m)-ndarray
                    - spins, float-(m, )-ndarray
                    - error, float-(m, )-ndarray
                    - tamp, float
                    - beta, float
                    - pump, float
                    - normalisation, float
                    - k, float
                    - c, float
                    - nl_func, pyfunc
                    - time_diff, float
                    - limit_func, pyfunc
    :return: updated spins and error terms: float-(m, )-ndarray, float-(m, )-ndarray
    """
    global update_func
    if pars['error_control'] == 'CAC':
        update_func = CAC_diff
    elif pars['error_control'] == 'CFC':
        update_func = CFC_diff
    elif pars['error_control'] == 'SFC':
        update_func = SFC_diff
    spin_step, error_step = update_func(**pars)
    return (limit_range(pars['spins'] + pars['time_diff']*spin_step, pars['tamp'], pars['limit_func']),
            pars['error'] + pars['time_diff']*error_step)

