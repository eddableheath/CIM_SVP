# Main file for CIM-SVP
# Author: Edmund Dable-Heath
"""
    Main simulation file for CIM
"""

import experiment_classes as CIM
import CIM_SVP_config as config
import numpy as np
import pandas as pd
np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 2000)


def main(pars, SVP=True, write=False):
    if SVP:
        experiment = CIM.CIM_SVP(**pars)
        if write:
            experiment.run(results_type='write')
            return 1
        results = experiment.run()
    else:
        experiment = CIM.CIM(**pars)
        results = experiment.run()

    print(results)


if __name__ == "__main__":
    parameters = config.pars
    main(parameters, SVP=False)

