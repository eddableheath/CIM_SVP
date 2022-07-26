# Generating Larger Lattices
# Author: Edmund Dable-Heath
"""
    Using the fpylll library to generate q-ary lattices and lll reduce them for preparation for CIM-SVP experiments.
"""

import numpy as np
from fpylll import IntegerMatrix, LLL


def generate_basis(dimension, q=13, lll=False):
    """
        Generates row representation basis matrix for a q-ary lattice where k=dim//2 and the quotient is 13 by default.
    :param dimension: lattice dimension, int
    :param q: lattice quotient - the q in q-ary, int
    :param lll: LLL flag, if True returns LLL reduced basis as well, bool
    :return:
            - lattice_basis, ndarray                                    - if LLL=False
            - (lattice_basis, LLL(lattice_basis)), (ndarray, ndarray)   - if LLL=True
    """
    a = IntegerMatrix.random(dimension, 'qary', k=dimension//2, q=q)
    z = np.zeros((dimension, dimension))
    _ = a.to_matrix(z)
    if not lll:
        return z
    else:
        b = LLL.reduction(a)
        y = np.zeros_like(z)
        _ = b.to_matrix(y)
        return z, y

