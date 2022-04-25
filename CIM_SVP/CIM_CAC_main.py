# Main run file for CIM CAC
# Author: Edmund Dable-Heath
# The main run file, where the actual experiment is run from. Can be embarrasingly parallelised in here as well.
#
# This is an implementation of https://arxiv.org/abs/2108.07369


import numpy as np
import CIM_CAC_sim as sim
import CIM_CAC_config as config
import CIM_CAC_experiment_classes as exp


def main(plots=False):
    """
        main function for running instantiation of experiment, including import of parameters from config file
    :param plots: (default = False) run experiment with plotting or not
    """
    experiment = exp.CIM_CAC_experiment(inters, config.tamp_start, config.tamp_raise, config.beta,
                                        config.pf_start, config.pf_raise, config.time_step, config.time_splits,
                                        config.no_repetitions, config.res_tol)
    experiment.run(plots)


def main_SVP(result_type, plots=False):
    """
        main function for running instantion of experiment, including import of parameter from the config file.

    :param result_type: what form are the results to be returned in:
                            'return': return a dataframe
                            'write': write the dataframe to the results folder

    :param plots: (default=False) also plopt spin trajectories
    """
    experiment = exp.CIM_CAC_SVP_experiment(config.lattice_basis, config.sitek, config.encoding, config.tamp_start,
                                            config.tamp_raise, config.beta, config.pf_start, config.pf_raise,
                                            config.time_step, config.time_splits, config.no_repetitions, config.res_tol,
                                            ground_energy=config.ground_energy)
    return experiment.run(result_type, plots)


if __name__ == "__main__":
    results = main_SVP('write')
    print(results)
    # inters = np.random.randn(11, 11)
    # for i in range(inters.shape[0]):
    #     inters[i, i] = 0.
    # print(inters)
    # main(Fa)

