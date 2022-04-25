# Construction of Ising interaction matrices
# Author: Edmund Dable-Heath
"""
    Given a lattice defined by basis B we want the ising coefficients for the QUBO representation. For the CIM we can
    only have coupling constants, no external field constants, so an auxillary site must be used who's state does not
    change.
"""

import numpy as np
np.set_printoptions(edgeitems=10, linewidth=180)


def poly_couplings(gramm, sitek):
    """
        Compute the polynomial encodings for the QUBO representation coupling strengths for the CIM.
    :param gramm: gramm matrix defining the problem, integer (m, m)-ndarray
    :param sitek: integer range to be explored, integer
    :return: - couplings array, float (n, n,)-ndarray
             - identity coefficient, float
    """
    qudits = gramm.shape[0]
    qubits_per_qudit = int(np.ceil(
        (np.sqrt(16 * sitek + 1) - 1) / 2
    ))
    qubits = qudits * qubits_per_qudit + 1
    jmat = np.zeros((qubits, qubits))
    ic = 0
    mu = np.ceil(qubits_per_qudit / 2) % 2

    for m in range(qudits):
        for l in range(qudits):
            for j in range(qubits_per_qudit):
                mj_qubit = (m * qubits_per_qudit) + j
                for k in range(qubits_per_qudit):
                    lk_qubit = (l * qubits_per_qudit) + k
                    coeff = 1/4 * (j+1)*(k+1) * gramm[m, l]
                    if mj_qubit == lk_qubit:
                        ic += coeff
                    else:
                        jmat[mj_qubit, lk_qubit] += coeff
        # Add in auxillary terms
        for j in range(qubits_per_qudit):
            mj_qubit = (m * qubits_per_qudit) + j
            coeff = mu * 1/4 * (j+1) * np.sum(gramm[m])
            jmat[mj_qubit, -1] = coeff
            jmat[-1, mj_qubit] = coeff

    ic += mu * 1/4 * np.sum(gramm)

    return jmat, ic


def bin_couplings(gramm, sitek):
    """
        Compute the binary encodings for the QUBO representation coupling strengths for the CIM.

        Note: Uses mapping 1 --> -1, 0 --> 1
    :param gramm: gramm matrix defining the problem, integer (m, m)-ndarray
    :param sitek: integer range to be explored, integer
    :return: - couplings array, float (n, n,)-ndarray
             - identity coefficient, float
    """
    qudits = gramm.shape[0]
    qubits_per_qudit = int(np.ceil(np.log2(sitek)) + 1)
    qubits = qudits * qubits_per_qudit + 1
    jmat = np.zeros((qubits, qubits))
    ic = 0

    for m in range(qudits):
        for l in range(qudits):
            for j in range(qubits_per_qudit):
                mj_qubit = (m * qubits_per_qudit) + j
                for k in range(qubits_per_qudit):
                    lk_qubit = (l * qubits_per_qudit) + k
                    coeff = 2**(j + k - 2) * gramm[m, l]
                    if mj_qubit == lk_qubit:
                        ic += coeff
                    else:
                        jmat[mj_qubit, lk_qubit] += coeff
        # Add in auxillary terms
        for j in range(qubits_per_qudit):
            mj_qubit = (m * qubits_per_qudit) + j
            coeff = 2**(j-2) * np.sum(gramm[m])
            jmat[mj_qubit, -1] = coeff
            jmat[-1, mj_qubit] = coeff

    ic += 1/4 * np.sum(gramm)

    return jmat, ic


def ham_couplings(gramm, sitek):
    """
            Compute the binary encodings for the QUBO representation coupling strengths for the CIM.
    :param gramm: gramm matrix defining the problem, integer (m, m)-ndarray
    :param sitek: integer range to be explored, integer
    :return: - couplings array, float (n, n,)-ndarray
             - identity coefficient, float
        """
    qudits = gramm.shape[0]
    qubits_per_qudit = 2 * sitek
    qubits = qudits * qubits_per_qudit
    jmat = np.zeros((qubits, qubits))
    ic = 0

    for m in range(qudits):
        for l in range(qudits):
            for j in range(qubits_per_qudit):
                mj_qubit = (m * qubits_per_qudit) + j
                for k in range(qubits_per_qudit):
                    lk_qubit = (l * qubits_per_qudit) + k
                    coeff = 1/4 * gramm[m, l]
                    if mj_qubit == lk_qubit:
                        ic += coeff
                    else:
                        jmat[mj_qubit, lk_qubit] += coeff

    return jmat, ic


# Testing
if __name__ == "__main__":
    basis = np.array([[1, 1],
                      [0, 1]])
    g = basis @ basis.T
    k = 10
    print(g)
    p_ij = poly_couplings(g, k)
    b_ij = bin_couplings(g, k)
    h_ij = ham_couplings(g, k)
    print(p_ij)
    print(b_ij)
    print(h_ij)
