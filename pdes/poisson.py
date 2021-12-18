from typing import Callable, Tuple

import numpy as np
import sympy
from scipy.sparse import bmat
from scipy.sparse.linalg.dsolve.linsolve import spsolve

from fem.boundary_conditions import (asm_dirichlet_boundary,
                                     asm_neumann_boundary)
from fem.error import error
from fem.fefunction import FEFunction
from fem.finite_elements import PElement
from fem.function_space import FEFunctionSpace
from fem.mesh import SimplexMesh

BoundaryCondition = Tuple[Callable, FEFunctionSpace]


def poisson(fs: FEFunctionSpace,
            fx: Callable,
            db: BoundaryCondition,
            nb: BoundaryCondition = None):
    L, M = fs.asm_stiff_mass()

    R, _, db_repr = asm_dirichlet_boundary(db, fs.dim)

    nb_repr = asm_neumann_boundary(nb) if nb is not None else np.zeros(fs.dim)

    # Readablity considered in favor of speed. For faster version see commit 56b7996
    f_int = M@f.coefficients
    sys = bmat([[L, R], [R.T, None]], format='csr')

    rhs = np.concatenate((f_int + nb_repr, db_repr))
    x = spsolve(sys, rhs)

    solution = FEFunction(fs)
    solution.coefficients = x[:fs.dim]

    return solution


if __name__ == '__main__':
    np.set_printoptions(precision=2, linewidth=178)
    np.seterr(divide='ignore', invalid='ignore')

    # Parameters to control regularity of the solution
    a = 1.1
    b = 0.

    x, y = sympy.symbols('x y')

    # Create a piecewise defined function with a kink along the x=0 line
    s_1_e = sympy.cos(np.pi * y / 2.)
    s_2_e = sympy.cos(np.pi * y / 2.) + (x - b * (y+1))**a

    s_1xx = sympy.diff(s_1_e, x, x)
    s_1yy = sympy.diff(s_1_e, y, y)

    s_2xx = sympy.diff(s_2_e, x, x)
    s_2yy = sympy.diff(s_2_e, y, y)

    l_1 = sympy.lambdify((x,y), (s_1xx + s_1yy))
    l_2 = sympy.lambdify((x,y), (s_2xx + s_2yy))
    s_1 = sympy.lambdify((x,y), s_1_e)
    s_2 = sympy.lambdify((x,y), s_2_e)

    def s(_x):
        x = _x[0]
        y = _x[1]
        fx = s_1(x,y) if x <= b*(y+1) else s_2(x,y)

        if np.isnan(fx):
            print("nan: {}, {}".format(x,y))
        return fx if not (np.isnan(fx) or np.isinf(fx)) else 0.

    def f(_x):
        x = _x[0]
        y = _x[1]
        if x < b*(y+1): fx = l_1(x,y)
        else: fx = l_2(x,y)

        if np.isnan(fx):
            print("nan: {}, {}".format(x,y))

        return -fx if not (np.isnan(fx) or np.isinf(fx)) else 0.


    d = 1

    errors = []
    for n in range(5, 40, 5):
        m = SimplexMesh.Create_2d_unit_square_structured(n, True)
        # m = import_gmsh('./lshape.msh')

        fs = FEFunctionSpace(m, PElement(d, m.element))

        db_fs = FEFunctionSpace(m.boundary_mesh,
                                PElement(d, m.element.lower_element))

        u = poisson(
            fs,
            f,
            (s, db_fs)
        )

        e_l2 = error(s, u, lambda x,y: np.abs(x-y)**2)**0.5
        print(e_l2)

        errors.append(e_l2)
        u.plot(1)
