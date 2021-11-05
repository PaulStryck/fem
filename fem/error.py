import numpy as np

from fem.integration import gauss_legendre_quadrature


def error(u, v, f):
    '''
    :param v: must be a FEFunction
    :param u: python function, range of u and v must be of the same dimension
    Integral of f(u(x), v(x)) on Ω
        Where Ω is the domain of v
    :param f: python function taking two arguments mapping to 1D
    '''
    quadrature = gauss_legendre_quadrature(v.fs.element.cell.dim,
                                           36)

    phi = v.fs.element.phi_eval(quadrature.points)
    integral = []
    for e, nodes in v.fs.elements:
        f_i     = v.fs.element.cell.affine_transform(e)
        detj, _ = v.fs.element.cell.affine_transform_jacobian(e)

        u_f_i_x = np.array(list(
            map(u, map(f_i, quadrature.points))
        ))

        v_f_i_x = np.dot(v.coefficients[nodes], phi.T)
        f_x = np.array(list(map(f, u_f_i_x, v_f_i_x)))
        val = np.dot(f_x, quadrature.weights) * detj
        integral.append(val)

    return sum(integral)
