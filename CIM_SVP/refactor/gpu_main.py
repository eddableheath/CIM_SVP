# Main file for GPU CIM SVP
# Author: Edmund Dable-Heath
"""
    Main simultation file for GPU implementation of CIM.

    Includes basis generation by fpylll library.
"""

import gpu_experiment_classes as CIM
import CIM_SVP_config as config
import numpy as np
import pandas as pd
from multiprocessing import Pool
from fpylll_latt_gen import generate_basis


def mp_run(pars, i):
    pars['repears'] = 1
    pars['seed'] = 12345*i
    experiment = CIM.CIM_SVP(**pars)
    return experiment.run()


def main(pars):
    basis, lll_basis = generate_basis(pars['dimension'], lll=True)
    np.savetxt('run_data/basis.csv', basis, delimiter=',', fmt='%5.0f')
    np.savetxt('run_data/lll.csv', lll_basis, delimiter=',', fmt='%5.0f')

    cores = pars['cores']

    # Regular run
    pars['basis'] = basis
    pool = Pool(cores)
    results = [
        pool.apply(
            mp_run,
            args=(pars, i)
        )
        for i in range(pars['repeats'])
    ]
    results = pd.concat(results)
    results.to_csv(f"{pars['path']}/qary.csv")
    pool.join()
    pool.close()

    # LLL run
    pars['basis'] = lll_basis
    pool = Pool(cores)
    lll_results = [
        pool.apply(
            mp_run,
            args=(pars, i)
        )
        for i in range(pars['repeats'])
    ]
    lll_results = pd.concat(results)
    results.to_csv(f"{pars['path']}/lll.csv")
    pool.join()
    pool.close()


if __name__=="__main__":
    parameters = config.pars
    main(parameters)
