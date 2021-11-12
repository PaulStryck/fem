from typing import Callable, Optional, Tuple

import numpy as np
import numpy.typing as npt
import sympy
from scipy.sparse import bmat
from scipy.sparse.coo import coo_matrix
from scipy.sparse.linalg.dsolve.linsolve import spsolve

from fem.boundary_conditions import (asm_dirichlet_boundary,
                                     asm_neumann_boundary)
from fem.error import error
from fem.fefunction import FEFunction
from fem.finite_elements import PElement
from fem.function_space import FEFunctionSpace
from fem.integration import gauss_legendre_quadrature
from fem.mesh import SimplexMesh

BoundaryCondition = Tuple[Callable, FEFunctionSpace]


def poisson(fs: FEFunctionSpace,
            f: FEFunction,
            db: BoundaryCondition,
            nb: BoundaryCondition = None):
    L, M = fs.asm_stiff_mass()

    R, _, db_repr = asm_dirichlet_boundary(db, fs.dim)

    nb_repr = asm_neumann_boundary(nb) if nb is not None else np.zeros(fs.dim)

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
    # a = 10
    # # RHS
    # def f(_x: npt.NDArray):
    #     x = _x[0]
    #     y = _x[1]
    #     s = 16**a
    #     s *= a*(-(x-1)*x)**(a - 2)
    #     s *= (-(y - 1)*y)**(a - 2)
    #     s *= (x**4 * (a*(1 - 2*y)**2 - 2*y**2 + 2*y - 1)
    #           - 2*x**3 * (a*(1 - 2*y)**2 - 2*y**2 + 2*y - 1)
    #           + x**2 * (a * (2*y**2 - 2*y + 1)**2 -
    #                     2*y**4 + 4*y**3 - 4*y**2 + 2*y - 1)
    #           - 2*(2*a - 1)*x*(y - 1)**2*y**2 + (a - 1)*(y - 1)**2*y**2)
    #     return -s

    # def s(_x):
    #     x = _x[0]
    #     y = _x[1]
    #     s = 2**(4*a)
    #     s *= x**a
    #     s *= (1 - x)**a
    #     s *= y**a
    #     s *= (1 - y)**a
    #     return s

    # a = 0.6
    # def f(x):
    #     if(x[0] == 0): return 10e8

    #     return (a - 1)*a*x[0]**(a - 2)

    # s = lambda x: x[0]**a

    a = 50
    xc = np.array([-0.05,-0.05])
    r0 = 0.7
    x, y = sympy.symbols('x y')
    expr = sympy.atan(a*(sympy.sqrt((x-xc[0])**2+(y-xc[1])**2) - r0))

    dxx = sympy.diff(expr, x, x)
    dyy = sympy.diff(expr, y, y)

    f_dxx = sympy.lambdify((x,y), dxx)
    f_dyy = sympy.lambdify((x,y), dyy)

    def f(_x):
        x = _x[0]
        y = _x[1]
        fx = f_dxx(x,y)
        fy = f_dyy(x,y)

        if np.isnan(fx):
            fx = 0

        if np.isnan(fy):
            fy = 0

        return -(fx+fy)

    # def f(_x):
    #     x = _x[0]
    #     y = _x[1]
    #     r = np.linalg.norm(_x-xc)
    #     g = a*(r-r0)
    #     gx = a*(x-xc[0])/r
    #     gxx = (a*r - a*(x-xc[0])**2)/(r**3)
    #     gy = a*(y-xc[1])/r
    #     gyy = (a*r - a*(y-xc[1])**2)/(r**3)

    #     sxx = (gxx*(g**2+1) - 2*g*(gx**2)) / ((gxx**2+1)**2)
    #     syy = (gyy*(g**2+1) - 2*g*(gy**2)) / ((gyy**2+1)**2)
    #     return sxx+syy

    s = lambda x: np.arctan(a * (np.linalg.norm(x - xc)-r0))


    ns = range(5, 65+1, 5)
    ns = [20]
    d = 3
    # for n in range(2,30,2):
    for n in ns:
        m = SimplexMesh.Create_2d_unit_square_structured(n)

        fs = FEFunctionSpace(m, PElement(d, m.element))
        f_interp = FEFunction(fs)
        f_interp.interpolate(f)

        db_fs = FEFunctionSpace(m.boundary_mesh,
                                PElement(d, m.element.lower_element))

        u = poisson(
            fs,
            f_interp,
            (s, db_fs)
        )

        e_l2 = error(s, u, lambda x,y: np.abs(x-y)**2)**0.5
        print(e_l2)
        # plot_mesh(m, True)

        sol = FEFunction(fs)
        sol.interpolate(f)
        sol.plot(1)
        # u.plot(1)

