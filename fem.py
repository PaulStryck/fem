from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix

from mesh import Face, Mesh


def asm_mass(m: Mesh):
    _i = np.zeros(len(m.faces) * 3**2, int)
    _j = np.zeros(len(m.faces) * 3**2, int)
    _data = np.zeros(len(m.faces) * 3**2)

    # mass matrix on the reference triangle
    ref_mass = np.array([[1/12, 1/24, 1/24],
                          [1/24, 1/12, 1/24],
                          [1/24, 1/24, 1/12]])

    # iterate over all faces
    n = 0
    for t, ind in m:
        # compute local mass matrix for each face
        detj, _ = jacobian(t)

        # scalar multiplication to map reference mass matrix
        # to deformed triangle
        local_mass = ref_mass * detj

        # map local mass matrix to global mass matrix
        for k,l in np.ndindex(local_mass.shape):
            _i[n] = ind[k]
            _j[n] = ind[l]

            _data[n] = local_mass[k,l]
            n += 1

    return coo_matrix((_data, (_i, _j)))


def asm_system(m: Mesh):

    _i = np.zeros(len(m.faces) * 3**2, int)
    _j = np.zeros(len(m.faces) * 3**2, int)
    _data = np.zeros(len(m.faces) * 3**2)

    b = np.zeros(len(m.vertices), float)

    n = 0

    for t, ind in m:
        detj, jt = jacobian(t)

        # grad*grad helper matrix (half part under the integral)
        gg = np.array([jt@dphi(1), jt@dphi(2), jt@dphi(3)])

        elemStiff = gg@gg.T
        elemStiff *= 0.5 * detj

        for k,l in np.ndindex(elemStiff.shape):
            _i[n] = ind[k]
            _j[n] = ind[l]

            _data[n] = elemStiff[k,l]
            n += 1

    return coo_matrix((_data, (_i, _j)))


def dphi(i):
    if i == 1:
        return np.array([-1.,-1.], float)
    if i == 2:
        return np.array([1., 0.], float)
    if i == 3:
        return np.array([0., 1.], float)

    return np.array([0., 0.], float)


def jacobian(t: Face) -> Tuple[float, npt.NDArray]:
    j = np.array([[t[1][0] - t[0][0], t[1][1] - t[0][1]],
                  [t[2][0] - t[0][0], t[2][1] - t[0][1]]], float)

    try:
        detj = np.linalg.det(j)
        j = np.linalg.inv(j)
    except Exception as e:
        print(t)
        raise e

    return detj, j
