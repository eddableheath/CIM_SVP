# Qubit Bound Computations
# Author: Edmund Dable-Heath
"""
    Using lemma one from Martin Albrecht's VQE paper this will compute the various qubit bounds for the three encodings:
                - binary
                - polynomial
                - hamming
    The polynomial and hamming bounds scale horrendously so an upper limit will be given based on the computation power.
"""

import numpy as np


def dual_basis(basis):
    """
        Compute the related dual basis for a given lattice basis (non-singular, full rank).
    :param basis: lattice basis, (m, m)-ndarray
    :return: dual basis, (m, m)-ndarray
    """
    basis = basis.T
    dual = basis @ np.linalg.inv(basis.T @ basis)
    return dual.T


def orthog_defect(basis):
    """
       Compute the orthogonality defect for a given basis
    :param basis: lattice basis, (m, m)-ndarray
    :return: orthogonality defect, float
    """
    return np.prod(np.linalg.norm(basis, axis=1)) / np.linalg.det(basis)


def bin_bound(basis):
    """
        Binary encoding qubit bound
    :param basis: lattice basis, (m, m)-ndarray
    :return: qubit bound, int
    """
    dim = basis.shape[0]
    return int(np.ceil(2*dim + np.log2(orthog_defect(dual_basis(basis))) + (dim/2)*np.log2(dim / (2*np.pi*np.e))))


def poly_bound(basis):
    """
        Polynomial encoding qubit bound
    :param basis: lattice basis, (m, m)-ndarray
    :return: qubit bound, int
    """
    dim = basis.shape[0]
    dbasis = dual_basis(basis)
    det = np.linalg.det(basis)
    s = 0
    for d in dbasis:
        s += np.sqrt(16 * (dim / (2*np.pi*np.e))**(1/dim)*det**(1/dim)*np.linald.norm(d) + 1)
    return int(np.ceil(0.5*s - 0.5*dim))


def hamming_bound(basis):
    """
        Hamming encoding qubit bound
    :param basis: lattice basis, (m, m)-ndarray
    :return: qubit bound, int
    """
    dim = basis.shape[0]
    dbasis = dual_basis(basis)
    return 2*(dim/(2*np.pi*np.e))**(1/dim) * np.linalg.det(basis)**(1/dim) * np.sum(np.linalg.norm(dbasis, axis=1))


