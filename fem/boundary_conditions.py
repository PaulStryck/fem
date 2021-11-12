from typing import Callable, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy.sparse.coo import coo_matrix

from fem.fefunction import FEFunction
from fem.function_space import FEFunctionSpace

BoundaryCondition = Tuple[Callable, FEFunctionSpace]


def asm_dirichlet_boundary(
    bc: Optional[BoundaryCondition],
    outer_dim: int
):
    if bc is None:
        return None, None

    f, fs = bc
    b_f = FEFunction(fs)
    b_f.interpolate(f)

    l = fs.dim
    R = coo_matrix(
        (np.ones(l), (b_f.embedded_coeffs_indices, np.arange(l))),
        shape=(outer_dim, l)
    )

    return R, b_f.embedded_coeffs_indices, b_f.embedded_coeffs_values


def asm_neumann_boundary(bc: BoundaryCondition) -> npt.NDArray:
    f, fs = bc

    # compute projection of BC Function
    _f = FEFunction(fs)
    _f.interpolate(f)

    _, M = fs.asm_stiff_mass(stiff=False, mass=True)

    if M is None:
        raise ValueError()

    # Filtering, in case its a SubMesh
    f_repr = np.zeros(M.shape[0])
    f_repr[_f.embedded_coeffs_indices] = _f.embedded_coeffs_values

    return M@f_repr
