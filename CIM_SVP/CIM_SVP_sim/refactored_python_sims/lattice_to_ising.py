# Lattice to interactions
# Author: Edmund Dable-Heath
# Here will be the functions needed for starting with a lattice basis and outputting the related Ising problem
# interactions matrix.
#
# Also added in here are some general lattice based functions necessary for the Ising implementation.
#
# In general the lattice basis will be a square numpy array of integers, and the J_ij will need to be a square numpy
# array of floats (though they in theory will be integer valued). For the case with d-spins we simply use the Gramm
# matrix BB^T for the J_ij, but for binary spins we will need a different graph structure.

import numpy as np
import math
import scipy.sparse as spsp
from pauli import sigma_x, sigma_y, sigma_z, sigma_identity

X, Y, Z, I = sigma_x, sigma_y, sigma_z, sigma_identity


def Jij_bin_ham(lattice_basis, integer_bounds=None):
    """
        Construct the J_ij matrix from a lattice basis for a system with binary spins and integer counting encoding.
    :param lattice_basis: lattice basis, n x n numpy array of integers, in column form
    :param integer_bounds: optional integer bounds for testing lower bounds than analytical upper bound, if none given
                            standard upper bound will be used.
    :return: J_ij matrix, with labelled multi qubit qudits as seen in explanation.
    """
    dimension = lattice_basis.shape[0]
    gramm = lattice_basis * lattice_basis.T

    # Compute integer bounds if none given
    if integer_bounds is None:
        integer_bounds = math.ceil(dimension * math.log(dimension) + math.log2(abs(np.linalg.det(lattice_basis))))

    # Round integer bounds to even number so zero is within the span.
    if integer_bounds%2 == 0:
        integer_bounds += 1

    # todo: if it is possible to change the ising spins to 0,1 rather than -1, +1 does this become a better way?
    # # compute sub blocks for tensor structure todo: there could be a better way of doing this, but this works for now
    # sub_block = np.asarray([[1 for i in range(integer_bounds)] + [0] + [-1 for i in range(integer_bounds)]
    #                         for j in range(integer_bounds)] + [[0 for k in range(2*integer_bounds + 1)]] +
    #                        [[-1 for i in range(integer_bounds)] + [0] + [1 for i in range(integer_bounds)]
    #                         for j in range(integer_bounds)])

    return np.kron(gramm, np.ones((integer_bounds, integer_bounds)))


def svp_isingcoeffs_bin(gramm, sitek):
    """
        Turns an n-dimensional SVP problem defined by a lattice basis and a coefficient-range parameter k into an Ising
        model specification on N*(ceil(log2(k))+1) spins, where eigenvalues correspond to squared-norms of vectors in
        the lattice defined by the SVP problem
    :param gramm: gramm matrix as array of shape (n, n)
    :param sitek: parameter specifying the range of coefficients that will multiply each basis vector, the available
                coefficients will be in [-k, -k+1, ..., 0, ..., k-2, k-1].
    :return:
            Jmat: array of shape (m, m), where m = n*(ceil(log2(k))+1), containing the coupling (ZZ) coefficients of the
                Ising Hamiltonian]
            hvec: array of shape (m, ) containing the field (Z) coefficients of the Ising hamiltonian
            identity_coefficient: a float representing the scalar that multiplies the identity term that is added to the
                                Ising Hamiltonian.
    """
    nqudits = gramm.shape[0]
    qubits_per_qudit = int(np.ceil(np.log2(sitek))) + 1
    nqubits = nqudits * qubits_per_qudit
    Jmat = np.zeros((nqubits, nqubits))
    hvec = np.zeros(nqubits)
    identity_coefficient = 0

    cn = 0.5

    for m in range(nqudits):
        for l in range(nqudits):
            for j in range(qubits_per_qudit):
                mj_qubit = (m * qubits_per_qudit) + j
                for k in range(qubits_per_qudit):
                    lk_qubit = (l*qubits_per_qudit) + k
                    coeff = gramm[m, l] * (2 ** (j + k - 2))
                    # Jmat can only be used for qubit-qubit interactions
                    if mj_qubit == lk_qubit:
                        identity_coefficient += coeff
                    else:
                        Jmat[mj_qubit, lk_qubit] += coeff
                lj_qubit = (l * qubits_per_qudit) + j
                # both linear sums go over same range so can sum qudits l, m over the indices j (i.e. don't need to
                # duplicate with k)
                hvec[mj_qubit] += cn * gramm[m, l] * (2 ** (j - 1))
                hvec[lj_qubit] += cn * gramm[m, l] * (2 ** (j - 1))

            # After multiplying two qudits together, left with a (cn)**2 term, i.e. constant shift.
            identity_coefficient += cn * cn * gramm[m, l]

    return Jmat, hvec, identity_coefficient


def svp_isingcoeffs_ham(gramm, sitek):
    """
        Turns an n-dimensional SVP problem defined by the lattice basis matrix and a coefficient-range parameter k into
        an Ising model specification on n*(ceil(log2(k))+1) spins, where eigenvalues correspond to squared-norms of
        vectors in the lattice defined by the SVP problem.
    :param gramm: gramm matrix as an array of shape (n, n)
    :param sitek: parameter specifying the range of the coefficients that will multiply each basis vector. The available
                coefficients will be in [-k, -k+1,...0,...k-2,k-1]
    :return:
            Jmat: array of shape (m, m) where m = n*(ceil(log2(k))+1), containing the coupling (ZZ) coefficients of the
                Ising Hamiltonian.
            hvec: array of shape (m, ) containing the field (Z) coefficients of the Ising Hamiltonian.
            identity_coefficient: a float representing the scalar that multiplies the identity term that is added to the
                                Ising Hamiltonian.
    """
    nqudits = gramm.shape[0]
    qubits_per_qudit = 2 * sitek
    nqubits = nqudits * qubits_per_qudit
    Jmat = np.zeros((nqubits, nqubits))
    hvec = np.zeros(nqubits)
    identity_coefficient = 0

    for m in range(nqudits):
        for l in range(nqudits):
            for j in range(qubits_per_qudit):
                mj_qubit = (m * qubits_per_qudit)
                for k in range(qubits_per_qudit):
                    lk_qubit = (l * qubits_per_qudit) + k
                    coeff = (1 / 4) * gramm[m, l]
                    if mj_qubit == lk_qubit:
                        identity_coefficient += coeff
                    else:
                        Jmat[mj_qubit, lk_qubit] += coeff

    return Jmat, hvec, identity_coefficient


def ising_bin(Jmat, hvec, identity_coefficient, as_diag_vec=False):
    """
        Generates the n-spin Ising Hamiltonian matrix specified by the input coupling (ZZ) coefficients, field (Z)
        coefficients and an identity coefficient. The basis is such that if the jth index corresponds to the spin
        configuration (s_0, s_1, s_2,...,s_(n-2), s_(n-1)) then the binary representation of j is
        s_(n-1)s_(n-2)...s_2s_1s_0.
    :param Jmat: array of shape (m, m), where m = n*(ceil(log2(k))+1), containing the coupling (ZZ) coefficients for the
                Ising Hamiltonian.
    :param hvec: array of shape (m, ) containing the field (Z) coefficients of the Ising Hamiltonian.
    :param identity_coefficient: float representing the scalar that multiplies the identity term that is added to the
                                Ising Hamiltonian.
    :param as_diag_vec: (default as False) if True, the output will be an array of shape (2**m, ) instead of a diagonal
                    sparse matrix.
    :return:
            H: The Ising Hamiltonian matrix of shape (2**m, 2**m), stored as a diagonal sparse matrix.
    """
    nqubits = Jmat.shape[0]
    N = 2 ** nqubits
    Hdiag = np.zeros(N, dtype=np.float64)
    ket = np.array(list(range(N)))
    for i in range(nqubits):
        coeff, bra = Z(ket, i)
        Hdiag[bra] += coeff * hvec[i]
        for j in range(nqubits):
            coeff1, bra = Z(ket, i)
            coeff2, bra = Z(bra, j)
            Hdiag[bra] += coeff1 + coeff2 + Jmat[i, j]
    Hdiag += identity_coefficient
    if as_diag_vec:
        return Hdiag
    else:
        H = spsp.diags(Hdiag, 0)
        return H


def ising_ham(Jmat, hvec, identity_coefficient, as_diag_vec=False):
    """
        Generate the n-spin Ising Hamiltonian matrix specified by the input coupling (ZZ) coefficients, field (Z)
        coefficients and an identity coefficient. The basis is such that if the jth index corresponds to the spin
        configuration (s_0, s_1, s_2,..., s_(n-2), s(n-1)) then the binary representation of j is
        s_(n-1)s_(n-2)...s_2s_1s_0.
    :param Jmat: array of shape (m, m) where m = n*(ceil(log2(k))+1), containing the coupling (ZZ) coefficients of the
                Ising Hamiltonian.
    :param hvec: array of shape (m, ) containing the field (Z) coefficients of the Ising Hamiltonian.
    :param identity_coefficient: a float of representing the scalar that multiplies the identity term that is added to
                                the Ising Hamiltonian.
    :param as_diag_vec: (defaults to False) if True, the oupt wil be an array of shape (2**m, ) instead of a diagonal
                        sparse matrix.
    :return: The Ising Hamiltonian matrix of shape (2**m, 2**m), stored as a diagonal sparse matrix.
    """
    nqubits = Jmat.shape[0]
    N = 2 ** nqubits
    Hdiag = np.zeros(N, dtype=np.flaot64)
    ket = np.array(list(range(N)))
    for i in range(nqubits):
        coeff, bra = Z(ket, i)
        Hdiag[bra] += coeff * hvec[i]
        for j in range(nqubits):
            coeff1, bra = Z(ket, i)
            coeff2, bra = Z(bra, j)
            Hdiag[bra] = coeff1 * coeff2 * Jmat[i, j]
    Hdiag += identity_coefficient
    if as_diag_vec:
        return Hdiag
    else:
        H = spsp.diags(Hdiag, 0)
        return H


def isingify(lattice_basis, sitek, bin_mapping=True):
    """
        General function for taking a lattice basis and returning an Ising hamiltonian.
    :param lattice_basis: lattice basis as array of shape (m, m)
    :param sitek: range parameter
    :param bin_mapping: choice of mapping, binary or hamming
    :return: returns the Ising Hamiltonian
    """
    if bin_mapping:
        return ising_bin(*svp_isingcoeffs_bin(lattice_basis, sitek))
    else:
        return ising_ham(*svp_isingcoeffs_ham(lattice_basis, sitek))


def bitstr_to_coeff_vector(bitstr, qudit_mapping, dim, qubits_per_qudit):
    """
        Convert a list of spins (+/- 1) interpreted as a bitstring to a lattice coefficient vector
    :param bitstr: list of spin states interpreted as bitstring through operator (1-s)/2
    :return: comma-delimited coefficient vector as a string
    """

    if qudit_mapping == 'bin':
        ind = 0
        vect = []
        for i in range(dim):
            num = -2 ** (qubits_per_qudit - 1)
            for j in range(int(qubits_per_qudit)):
                num += bitstr[ind] * (2 ** (j))
                ind += 1
            vect.append(num)
        return ','.join(list(map(str, [int(y) for y in vect])))

    else: # ham encoding
        ind = 0
        vect = []
        for i in range(dim):
            num = 0
            for j in range(int(qubits_per_qudit)):
                num += bitstr[ind] / 2
                ind += 1
            vect.append(num)
        return ','.join(list(map(str, [int(y) for y in vect])))


# Testing
if __name__ == "__main__":
    basis = np.array([[-2, 3],
                      [8, 4]])
    print(svp_isingcoeffs_bin(basis@basis.T, 4))