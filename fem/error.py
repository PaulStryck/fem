import numpy as np

from .integration import gauss_legendre_quadrature


def error(u, v, f, grad=False):
    '''
    Compute ∫ f(u(x), v(x)) dx on Ω
    :param u: python function, range of u and v must be of the same dimension
    :param v: must be a :class FEFunction: with domain Ω
    :param f: python function taking two arguments mapping to 1D
    '''

    # Quadrature rule of order 36 where the spation dimension
    # matches the spation dimension of the reference element of the mesh
    q_dim = min([3, v.fs.element.dim**2])
    quadrature = gauss_legendre_quadrature(v.fs.element.cell.dim,
                                           q_dim)

    # Evaluate the local basis function at all quadrature points
    if grad:
        phi = v.fs.element.grad_phi_eval(quadrature.points)
    else:
        phi = v.fs.element.phi_eval(quadrature.points)

    # Store the intermediate values and compute sum afterwords to prevent
    # round off errors
    integral = []

    # Outer sum. Splitting Ω into mesh elements
    for e, nodes in v.fs.elements:
        # Mapping for reference elemnt to current mesh element
        F_i     = v.fs.element.cell.affine_transform(e)

        detj, jTinv = v.fs.element.cell.affine_transform_jacobian(e)

        # u(F(x)) for all quadrature points
        u_F_i_x = np.array(list(
            map(u, map(F_i, quadrature.points))
        ))

        if grad:
            v_F_i_x = v.coefficients[nodes] @ (phi @ jTinv.T)
        else:
            v_F_i_x = np.dot(v.coefficients[nodes], phi.T)

        # f(u(F(x)), v(F(x)))
        f_x = np.array(list(map(f, u_F_i_x, v_F_i_x)))

        # finally quadrature weights and jacobian determinant
        val = np.dot(f_x, quadrature.weights) * detj
        integral.append(val)

    return sum(integral)
