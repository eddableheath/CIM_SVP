# GPU Simulation Functions
# Author: Edmund Dable-Heath
"""
    Replicating the simulation functions for the CIM for GPU implementation.

    Requires:
        - PyTorch
        - Cuda availability.
"""

import numpy as np
import torch

# Global variables
global update_func


def ising_energy(state, couplings):
    """
        For a given state compute the Ising energy.

        N.B. this is without the adjusting identity coefficient required for the QUBO implementation.
    :param state: current spins (configured to {-1, +1} state by sign), float-(m, )-torch.tensor
    :param couplings: couplings matrix, float-(m, m)-torch.tensor
    :return: Ising energy of current state, float
    """
    return torch.matmul(torch.sign(state), torch.matmul(couplings, torch.sign(state)))


def limit_range(state, tamp, limit_func):
    """
        limit the range of the current spin state
    :param state: current spins, float-(m, m)-torch,tensor
    :param tamp: current t-amplitude, float
    :param limit_func: choice of limiting function, pyfunc
    :return: bounded spins in state, float-(m, )-torch.tensor
    """
    bound = limit_func(tamp)
    if bound is None:
        return state
    return torch.minimum(
        torch.maximum(
            state, torch.tensor([-bound for _ in range(state.shape[0])])
        ), torch.tensor([bound for _ in range(state.shape[0])])
    )


# Spin and error diffs for each error correcting CIM
def CAC_diff(**pars):
    """
       compute the simulation step for the CAC control.
    :param pars: dictionary of parameters for function:
                    - couplings, float-(m, m)-torch.tensor
                    - spins, float-(m, )-torch.tensor
                    - error, float-(m, )-torch.tensor
                    - tamp, float
                    - beta, float
                    - pump, float
                    - normalisation, float
    :return: timestep diff for spins and error terms: float-(m, )-torch.tensor, float-(m, )-torch.tensor
    """
    ising_sum = pars['normalisation'] * pars['error'] * torch.matmul(pars['couplings'], pars['spins'])
    return (-pars['spins']**3 + pars['pump']*pars['spins'] - ising_sum,
            -pars['beta']*pars['error']*(ising_sum**2 - pars['tamp']))


def CFC_diff(**pars):
    """
       compute the simulation step for the CFC control.
    :param pars: dictionary of parameters for function:
                    - couplings, float-(m, m)-torch.tensor
                    - spins, float-(m, )-torch.tensor
                    - error, float-(m, )-torch.tensor
                    - tamp, float
                    - beta, float
                    - pump, float
                    - normalisation, float
    :return: timestep diff for spins and error terms: float-(m, )-torch.tensor, float-(m, )-torch.tensor
    """
    ising_sum = pars['error']*pars['normalisation']*torch.matmul(pars['couplings'], pars['spins'])
    return (-pars['spins']**3 + pars['pump']*pars['spins'] - ising_sum,
            -pars['beta']*pars['error']*(ising_sum**2-pars['tamp']))


def SFC_diff(**pars):
    """
       compute the simulation step for the SFC control.
    :param pars: dictionary of parameters for function:
                    - couplings, float-(m, m)-torch.tensor
                    - spins, float-(m, )-torch.tensor
                    - error, float-(m, )-torch.tensor
                    - tamp, float
                    - beta, float
                    - pump, float
                    - normalisation, float
                    - k, float
                    - c, float
                    - nl_func, pyfunc
    :return: timestep diff for spins and error terms: float-(m, )-torch.tensor, float-(m, )-torch.tensor
    """
    ising_sum = pars['normalisation']*torch.matmul(pars['couplings'], pars['spins'])
    return (-pars['spins']**3 + pars['pump']*pars['spins'] -
            pars['nl_func'](pars['c']*ising_sum) - pars['k']*(ising_sum-pars['error']),
            -pars['beta']*pars['error']*(ising_sum**2-pars['tamp']))


def simulation_step(**pars):
    """
        single time step of the simulation
    :param pars: dictionary of parameters for function:
                    - error_control, str
                    - couplings, float-(m, m)-torch.tensor
                    - spins, float-(m, )-torch.tensor
                    - error, float-(m, )-torch.tensor
                    - tamp, float
                    - beta, float
                    - pump, float
                    - normalisation, float
                    - k, float
                    - c, float
                    - nl_func, pyfunc
                    - time_diff, float
                    - limit_func, pyfunc
    :return: updated spins and error terms: float-(m, )-torch.tensor, float-(m, )-torch.tensor
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


