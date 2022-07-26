# Main file for CIM-SVP
# Author: Edmund Dable-Heath
"""
    Main simulation file for CIM
"""

import experiment_classes as CIM
import CIM_SVP_config as config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sim_funcs import ising_energy
import multiprocessing as mp
np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 2000)


def mp_run(pars, i):
    pars['repeats'] = 1
    pars['seed'] = 12345*i
    experiment = CIM.CIM_SVP(**pars)
    return experiment.run()


def main(
        pars, SVP=True, write=False, plots=False, multiprocessed=False, update_basis=False
):
    if plots:
        pars['save_traj'] = True
    if not multiprocessed:
        if SVP:
            if update_basis:
                pars['update_basis'] = True
                experiment = CIM.CIM_SVP_basis_update(**pars)
                results, bases = experiment.run()
            else:
                experiment = CIM.CIM_SVP(**pars)
                if write:
                    experiment.run(results_type='write')
                    return 1
                results = experiment.run()
        else:
            experiment = CIM.CIM(**pars)
            results = experiment.run()
        print(bases)
        print(results)
        # print(experiment.couplings)
        print(np.linalg.norm(experiment.basis, axis=1))
        print(f"mean norm found: {results['norms'].mean(skipna=True)}")
        print(f"min norm found: {min(results['norms'])}")
        print(f"max norm found: {max(results['norms'])}")
        if plots:
            for i in range(experiment.repeats):
                # for basis in bases:
                spin_results = experiment.traj_spins[i]
                error_results = experiment.traj_error[i]
                ising_energies = np.asarray([ising_energy(spin_results[j], experiment.couplings)
                                             for j in range(spin_results.shape[0])])
                if SVP:
                    ising_energies += experiment.identity_coeff
                plt.title(f'Spin Trajectory for result {i}')
                x_axis = np.array(range(experiment.iters)) * experiment.timediff
                plt.ylim((-2.5, 2.5))
                plt.plot(x_axis, spin_results)
                plt.show()
                plt.close()

                # plt.title(f'Error Trajectory for result {i}')
                # x_axis = np.array(range(experiment.iters)) * experiment.timediff
                # plt.ylim((0, np.max(error_results)))
                # plt.plot(x_axis, error_results)
                # plt.show()
                # plt.close()
                #
                # plt.title(f'Residual Ising Energy Trajectory for result {i}')
                # x_axis = np.array(range(experiment.iters)) * experiment.timediff
                # plt.ylim((0, 2*np.median(ising_energies)))
                # plt.plot(x_axis, ising_energies)
                # plt.show()
                input('press enter to continue...')

    else:
        cores = pars['cores']
        pool = mp.Pool(cores)
        results = [
            pool.apply(
                mp_run,
                args=(pars, i)
            )
            for i in range(pars['repeats'])
        ]
        # print(results)
        results = pd.concat(results)
        # print(results)
        results.to_csv(f"{pars['path']}/test.csv")


if __name__ == "__main__":
    parameters = config.pars
    main(parameters, SVP=True, update_basis=True, plots=True, multiprocessed=True)

