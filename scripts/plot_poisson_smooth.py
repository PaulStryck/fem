import json
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import sympy
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fem.error import error
from fem.fefunction import FEFunction
from fem.finite_elements import PElement
from fem.function_space import FEFunctionSpace
from fem.mesh import SimplexMesh
from pdes.poisson import poisson

from .plot_mesh import plot_mesh


def plot_function(f, n, ax, cmapinterval=(0,1)):
        m = SimplexMesh.Create_2d_unit_square_structured(n)

        fs = FEFunctionSpace(m, PElement(d, m.element))
        f_interp = FEFunction(fs)
        f_interp.interpolate(f)

        f_interp.plot(ax, cmapinterval=cmapinterval)


if __name__ == '__main__':
    a = 5.

    # correct solution
    x, y = sympy.symbols('x y')
    s_e = 2**(4*a) * x**a * (1 - x)**a * y**a * (1 - y)**a

    # laplacian of s_e
    l_e = sympy.diff(s_e, x, x) + sympy.diff(s_e, y, y)

    sx = sympy.diff(s_e, x)
    sy = sympy.diff(s_e, y)

    # evaluatable functions for solution and laplacian
    s_ = sympy.lambdify((x,y), s_e)
    l_ = sympy.lambdify((x,y), l_e)

    sx_ = sympy.lambdify((x,y), sx)
    sy_ = sympy.lambdify((x,y), sy)


    # evaluatable functions catching NAN and INF values
    def s_py(x,y):
        fx = s_(x,y)

        return fx if not (np.isnan(fx) or np.isinf(fx)) else 0.

    s_np = lambda x: s_py(x[0], x[1])


    def g(_x):
        x = _x[0]
        y = _x[1]
        fx = sx_(x,y)
        fy = sy_(x,y)

        fx = fx if not (np.isnan(fx) or np.isinf(fx)) else 0.
        fy = fy if not (np.isnan(fy) or np.isinf(fy)) else 0.

        return np.array([fx, fy])

    def f_py(x,y):
        fx = l_(x,y)

        return -fx if not (np.isnan(fx) or np.isinf(fx)) else 0.
    f_np = lambda x: f_py(x[0], x[1])

    s = np.vectorize(s_py)
    f = np.vectorize(f_py)


    ns = range(4, 20+1, 2)
    d = 1
    f_name = 'errs_1_new.json'


    errs = None
    try:
        fp = open(f_name, 'r')
    except IOError:
        print("No errors found, generate new")
    else:
        with fp:
            errs = json.load(fp)

    if errs is not None:
        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(2, 4, figure=fig)

        ax_3d = fig.add_subplot(gs[0,0], projection='3d')
        ax_3d.set_title("Solution u")
        plot_function(s_np, 10, ax_3d)
        ax_sol  = fig.add_subplot(gs[0,1], projection='rectilinear')
        ax_rhs  = fig.add_subplot(gs[1,0], projection='rectilinear')

        for i, n in enumerate([4,6]):
            ax = fig.add_subplot(gs[0,i+2])
            ax.set_title('Regular Grid n = {}'.format(n))
            ax.tick_params(left=False,
                           bottom=False,
                           labelleft=False,
                           labelbottom=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            plot_mesh(ax,
                      SimplexMesh.Create_2d_unit_square_unstructured(n),
                      False)

        ax_errs = fig.add_subplot(gs[1,1:])

        ax_sol.set_title("Solution u")

        ax_rhs.set_title("RHS f", pad=3, y=1.000001)
        # ax_rhs.view_init(10, -60)
        x = np.linspace(0, 1, 500)
        y = np.linspace(0, 1, 500)
        X, Y = np.meshgrid(x, y)

        # ax_sol.contourf(X, Y, s(X,Y), levels=np.linspace(0,1,30))
        # ax_rhs.contourf(X, Y, f(X,Y), levels=np.linspace(-20,80,30))

        extent = np.min(X), np.max(X), np.min(Y), np.max(Y)

        im_s = ax_sol.imshow(s(X,Y),
                             cmap=plt.cm.viridis,
                             interpolation='nearest',
                             extent=extent)

        im_f = ax_rhs.imshow(f(X,Y), cmap=plt.cm.viridis, interpolation='nearest',
                           extent=extent)

        for ax, im in [(ax_sol, im_s), (ax_rhs, im_f)]:
            ax.tick_params(left=False,
                           bottom=False,
                           labelleft=False,
                           labelbottom=False)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("left", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax, orientation='vertical',
                         ticklocation='left')

        # plot_function(f, 10, ax_rhs, cmapinterval=(-10,80))
        ax_errs.loglog(errs['n'], errs['l2'], 'rx', label='L2 Error')
        ax_errs.loglog(errs['n'], errs['h1'], 'bx', label='H1 Error')
        ax_errs.legend()
        ax_errs.set_xlabel("n")
        ax_errs.set_ylabel("Error")

        plt.show()
        exit()



    if len(ns) <= 3:
        subplt = (1, len(ns))
    else:
        subplt = (int(np.floor(np.sqrt(len(ns)))),
                  int(np.ceil(np.sqrt(len(ns)))))

    errors_l2 = []
    errors_h1 = []
    for i, n in enumerate(ns):
        # m = SimplexMesh.Create_2d_unit_square_structured(n)
        m = SimplexMesh.Create_2d_unit_square_unstructured(n)

        fs = FEFunctionSpace(m, PElement(d, m.element))

        db_fs = FEFunctionSpace(m.boundary_mesh,
                                PElement(d, m.element.lower_element))

        u = poisson(
            fs,
            f_np,
            (s_np, db_fs)
        )

        e_l2 = error(s_np, u, lambda x,y: (np.abs(x-y)**2), grad=False)
        e_h1 = error(g, u, lambda x,y: np.linalg.norm(x-y)**2, grad=True)
        errors_l2.append(e_l2**0.5)
        errors_h1.append(e_h1**0.5)
        print("n = {}, H1_err = {}".format(n, errors_h1[-1]))


    with open(f_name, 'w') as fp:
        json_data = json.dump({
            'n': list(ns),
            'l2': errors_l2,
            'h1': errors_h1
        }, fp=fp)
