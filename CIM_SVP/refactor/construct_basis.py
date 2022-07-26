# Construct a basis from a (primitive) vector
# Author: Edmund Dable-Heath
"""
    Given a sampled vector from a lattice with a particular basis:

        1. Check it's primitive and make primitive if not.
        2. Construct a new basis from the primitive vector and the old basis, preserving any chosen basis vectors.
"""

import numpy as np
from math import gcd, hypot
from functools import reduce


def check_for_primitivity(vec: np.array) -> np.array:
    """
        Checks to see if the input vector is primitive, and if not returns an updated primitive vector from it.
    :param vec: input vector, int-(m, )-ndarray
    :return: primitive vector, int-(m, )-ndarray
    """
    while reduce(gcd, vec) > 1:
        vec = (vec / reduce(gcd, vec)).astype(int)
    return vec


def extended_euclid_gcd(a: int, b: int) -> list:
    """
    Returns a list `result` of size 3 where:
    Referring to the equation ax + by = gcd(a, b)
        result[0] is gcd(a, b)
        result[1] is x
        result[2] is y
    """
    s = 0; old_s = 1
    t = 1; old_t = 0
    r = b; old_r = a

    while r != 0:
        quotient = old_r//r
        old_r, r = r, old_r - quotient*r
        old_s, s = s, old_s - quotient*s
        old_t, t = t, old_t - quotient*t
    return [old_r, old_s, old_t]


def unimodular_2(x: int, y: int) -> (int, int, int, np.array):
    """
        Compute a 2 dimensional unimodular matrix from integer inputs
    :param x: integer 1, int
    :param y: integer 2, int
    :return: unimodular matrix, int-(2, 2)-ndarray
    """
    d, a, b = extended_euclid_gcd(x, y)
    return a, b, d, np.array([[x/d, -b],
                              [y/d, a]]),


def construct_basis(prim_vec: np.array, basis: np.array) -> np.array:
    """
        Construct a basis from the previous basis and a primitive vector.
    :param prim_vec: primitive vector with gcd = 1 across the vector
    :param basis: original basis
    :return: updated basis
    """
    q, r = np.linalg.qr(basis.T)
    z = np.eye(basis.shape[0])
    for j in range(basis.shape[0]-1, 0, -1):
        a, b, d, U = unimodular_2(prim_vec[j-1], prim_vec[j])
        prim_vec[j-1] = d
        r[0:j+1, j-1:j+1] = r[0:j+1, j-1:j+1] @ U
        z[:, j-1:j+1] = z[:, j-1:j+1] @ U
        if r[j, j] != 0:
            h = hypot(r[j-1, j], r[j, j])
            p = 1. / h
            c = abs(r[j-1, j]) * p
            s = np.sign(r[j-1, j])*p*r[j, j]
        else:
            c = 1.
            s = 0.
        G = np.array([[c, -s],
                      [s, c]])
        r[j-1:j+1, j-1:] = G @ r[j-1:j+1, j-1:]
    return basis.Ta @ z


# Basis Comparisons
def mean_comp(basis, vector):
    if np.linalg.norm(vector) < np.mean(np.linalg.norm(basis, axis=1)):
        return True
    else:
        return False


def max_comp(basis, vector):
    if np.linalg.norm(vector) < np.max(np.linalg.norm(basis, axis=1)):
        return True
    else:
        return False


def med_comp(basis, vector):
    if np.linalg.norm(vector) < np.median(np.linalg.norm(basis, axis=1)):
        return True
    else:
        return False


def min_comp(basis, vector):
    if np.linalg.norm(vector) < np.min(np.linalg.norm(basis, axis=1)):
        return True
    else:
        return False


# testing
if __name__ == "__main__":
    B = np.array([[3, 0, 15, -12],
                  [0, 4, 3, 8],
                  [28, -18, 9, 8],
                  [0, 0, 3, -4]])
    v = np.array([1, 2, 3, 5])
    print(v)
    new_v = check_for_primitivity(v)
    print(new_v)
    new_basis = construct_basis(new_v, B)
    print(new_basis)
    print(np.linalg.det(B))
    print(np.linalg.det(new_basis))
    if np.round(np.linalg.det(B)) == np.round(np.linalg.det(new_basis)):
        print('yes')
