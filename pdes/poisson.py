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
from fem.integration import gauss_legendre_quadrature
from fem.mesh import SimplexMesh

BoundaryCondition = Tuple[Callable, FEFunctionSpace]


def poisson(fs: FEFunctionSpace,
            fx: Callable,
            db: BoundaryCondition,
            nb: BoundaryCondition = None):
    L, _ = fs.asm_stiff_mass()

    R, _, db_repr = asm_dirichlet_boundary(db, fs.dim)

    nb_repr = asm_neumann_boundary(nb) if nb is not None else np.zeros(fs.dim)

    ######## TEST ##########
    numDataPts = fs.dim

    f_int = np.zeros(numDataPts, dtype=np.double)
    quadrature = gauss_legendre_quadrature(fs.element.cell.dim,
                                           fs.element.dim**2)
    Phi  = fs.element.phi_eval(quadrature.points)

    for e, ind in fs.elements:
        detj, _ = fs.element.cell.affine_transform_jacobian(e)
        F_i = fs.element.cell.affine_transform(e)

        fxs = np.array([fx(F_i(x)) for x in quadrature.points])

        loc_f_int = np.sum(
            np.multiply(np.multiply(fxs, quadrature.weights), Phi.T).T * detj,
            axis = 0
        )

        for k in range(len(loc_f_int)):
            f_int[ind[k]] += loc_f_int[k]

    ######## END TEST ##########

    # f_int = M@f.coefficients
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

    # a = 50
    # xc = np.array([-0.05,-0.05])
    # r0 = 0.7
    # x, y = sympy.symbols('x y')
    # expr = sympy.atan(a*(sympy.sqrt((x-xc[0])**2+(y-xc[1])**2) - r0))

    # dxx = sympy.diff(expr, x, x)
    # dyy = sympy.diff(expr, y, y)

    # f_dxx = sympy.lambdify((x,y), dxx)
    # f_dyy = sympy.lambdify((x,y), dyy)

    # def f(_x):
    #     x = _x[0]
    #     y = _x[1]
    #     fx = f_dxx(x,y)
    #     fy = f_dyy(x,y)

    #     if np.isnan(fx):
    #         fx = 0

    #     if np.isnan(fy):
    #         fy = 0

    #     return -(fx+fy)

    # s = lambda x: np.arctan(a * (np.linalg.norm(x - xc)-r0))
    # a = 2./3.
    # x, y = sympy.symbols('x y')

    # expr_r = (sympy.sqrt(x**2+y**2)**a)*sympy.sin(a*sympy.atan(y/x))
    # expr_l = (sympy.sqrt(x**2+y**2)**a)*sympy.sin(a*(sympy.atan(y/x)+np.pi))

    # dxx_r = sympy.diff(expr_r, x, x)
    # dxx_l = sympy.diff(expr_l, x, x)
    # dyy_r = sympy.diff(expr_r, y, y)
    # dyy_l = sympy.diff(expr_l, y, y)

    # f_dxx_r = sympy.lambdify((x,y), dxx_r)
    # f_dxx_l = sympy.lambdify((x,y), dxx_l)
    # f_dyy_r = sympy.lambdify((x,y), dyy_r)
    # f_dyy_l = sympy.lambdify((x,y), dyy_l)

    # def s(_x):
    #     x = _x[0]
    #     y = _x[1]
    #     r = np.linalg.norm(_x)
    #     if x == 0 and y == 0: t = 0
    #     elif x == 0 and y > 0: t = np.pi/2.
    #     elif x == 0 and y < 0: t = 3*np.pi/2.
    #     elif x < 0: t = np.arctan(y/x) + np.pi
    #     else: t = np.arctan(y/x)

    #     return r**a*np.sin(a*t)

    # def f(_x):
    #     x = _x[0]
    #     y = _x[1]
    #     if x >= 0:
    #         fx = f_dxx_r(x,y)
    #         fy = f_dyy_r(x,y)
    #     else:
    #         fx = f_dxx_l(x,y)
    #         fy = f_dyy_l(x,y)

    #     if np.isnan(fx):
    #         fx = 0

    #     if np.isnan(fy):
    #         fy = 0

    #     return -(fx+fy)

    a = 1.1
    b = 0.

    x, y = sympy.symbols('x y')
    s_1_e = sympy.cos(np.pi * y / 2.)
    s_2_e = sympy.cos(np.pi * y / 2.) + (x - b * (y+1))**a
    # s_1_e = 0. * x
    # s_2_e = x**a

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


    ns = range(12, 200+1, 12)
    ns = [20]
    d = 1
    # d = 3
    errors = []
    for n in ns:
        m = SimplexMesh.Create_2d_unit_square_structured(n)
        # m = SimplexMesh.Create_2d_refined(n)
        # m = SimplexMesh.Create_2d_unit_square_unstructured(n)
        # m = SimplexMesh.Create_2d_L(10, (2.5/2.)*np.pi)
        # m = import_gmsh('./lshape.msh')

        fs = FEFunctionSpace(m, PElement(d, m.element))
        f_interp = FEFunction(fs)
        # f_interp.interpolate(f)

        db_fs = FEFunctionSpace(m.boundary_mesh,
                                PElement(d, m.element.lower_element))

        u = poisson(
            fs,
            f,
            (s, db_fs)
        )

        e_l2 = error(s, u, lambda x,y: np.abs(x-y)**2)**0.5
        print(e_l2)
        # errors.append(e_l2)
        # plot_mesh(m, True)

        # sol = FEFunction(fs)
        # sol.interpolate(f)
        # sol.plot(1)
        u.plot(1)
    print(errors)
