# Main code playground for CIM SVP project
# Author: Edmund Dable-Heath
"""
    So we're putting together the CIM SVP simulation
"""

import numpy as np
import isingify as li


def interactions(gramm, sitek):
    """
        Compute interaction matrix for given gramm matrix (prob spec) and integer range (sitek)
    :param gramm: gramm matrix
    :param sitek: integer range
    :return: interactions matrix and constant identity shift
    """
    qudits = gramm.shape[0]
    qubits_per_qudit = int(np.ceil(
        (np.sqrt(16 * sitek + 1) - 1) / 2
    ))
    qubits = qudits * qubits_per_qudit
    print(f'qpq: {qubits_per_qudit}')
    jmat = np.zeros((qubits, qubits))
    ic = 0
    mu = np.ceil(qubits_per_qudit / 2) % 2
    print(mu)

    for m in range(qudits):
        for l in range(qudits):
            for j in range(qubits_per_qudit):
                mj_qubit = (m * qubits_per_qudit) + j
                for k in range(qubits_per_qudit):
                    lk_qubit = (l * qubits_per_qudit) + k
                    coeff = (j+1)*(k+1) + mu*((j+1) + (k+1) + 1)
                    coeff *= 1/4*gramm[m, l]
                    if mj_qubit == lk_qubit:
                        ic += coeff
                    else:
                        jmat[mj_qubit, lk_qubit] = coeff
    return jmat, ic


def binteractions(gramm, sitek):
    """
        Compute interaction matrix for given gramm matrix (prob spec) and integer range (sitek)
    :param gramm: gramm matrix
    :param sitek: integer range
    :return: interactions matrix and constant identity shift
    """
    qudits = gramm.shape[0]
    qubits_per_qudit = int(np.floor(np.log2(sitek))+1)
    qubits = qudits * qubits_per_qudit
    jmat = np.zeros((qubits, qubits))
    ic = 0

    for m in range(qudits):
        for l in range(qudits):
            for j in range(qubits_per_qudit):
                mj_qubit = (m * qubits_per_qudit) + j
                for k in range(qubits_per_qudit):
                    lk_qubit = (l * qubits_per_qudit) + k
                    coeff = 2**(j+k) + 2**j + 2**k + 1
                    coeff *= (1/4)*gramm[m, l]
                    if mj_qubit == lk_qubit:
                        ic += coeff
                    else:
                        jmat[mj_qubit, lk_qubit] = coeff
    return jmat, ic


def ising_energy(vec, inters):
    s = 0
    for i in range(vec.shape[0]):
        for j in range(vec.shape[0]):
            s += vec[i] * vec[j] * inters[i, j]
    return s


def main():
    inters, id_con = li.poly_couplings(
        pars['basis']@pars['basis'].T,
        pars['int_range']
    )
    print(inters)
    print(id_con)
    gramm = pars['basis'] @ pars['basis'].T
    qubits_per_qudit = int(np.ceil(
        (np.sqrt(16 * pars['int_range'] + 1) - 1) / 2
    ))
    print(qubits_per_qudit)
    # qubits_per_qudit = int(np.ceil(np.log2(pars['int_range'])) + 1)
    mu = int(np.ceil(qubits_per_qudit/2)) % 2
    print(mu)
    rand_vect = np.random.randint(0, 2, inters.shape[0])
    rand_vect[-1] = 1
    print(rand_vect)
    rand_vect = (2*rand_vect) - 1
    print(rand_vect)
    print(ising_energy(rand_vect, inters))
    print(ising_energy(rand_vect, inters) + id_con)
    vect = np.zeros(gramm.shape[0])
    for i in range(gramm.shape[0]):
        vect[i] = int(np.sum(
            rand_vect[i*qubits_per_qudit:(i+1)*qubits_per_qudit] * (np.arange(1, qubits_per_qudit+1)/2)
        )+(mu/2))
    # for i in range(gramm.shape[0]):
    #     vect[i] = int(np.sum(
    #         -rand_vect[i*qubits_per_qudit:(i+1)*qubits_per_qudit] * 2**np.arange(qubits_per_qudit)
    #     )) - 2**qubits_per_qudit
    print(vect)
    latt_vect = pars['basis'].T @ vect
    print(latt_vect)



if __name__ == '__main__':
    pars = {
        'basis': np.array([[1, 1], [0, 1]]),
        'int_range': 6
    }
    main()
