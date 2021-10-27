import numpy as np

from fem import jacobian, phi
from integration import gauss_legendre_quadrature


def error_norm(u, uh_m, norm='l2'):
    '''
    :param u: must be a function of type (number, number) -> number
    :param uh_m: must be a MeshFunction (u_h, m)
    '''
    # unpack coefficients and mesh from MeshFunction
    (u_h, m) = uh_m

    # store the integral of all faces to compute sum later in a more stable way
    fragments = np.zeros(len(m.faces))

    # integrate on 2d simplex with 6th degree gauss quadrature
    integrator = gauss_legendre_quadrature(2, 6)

    # iterate over all faces
    for i, (t, ind) in enumerate(m):
        # Build transfrom mapping the unit simplex to a face
        # F(x) = Bx + d
        B = np.array([[t[1][0] - t[0][0], t[2][0] - t[0][0]],
                      [t[1][1] - t[0][1], t[2][1] - t[0][1]]], float)
        d = np.array([t[0][0], t[0][1]])
        detj = np.linalg.det(B)

        # different integrands depending on norm
        # the expression u_h.take(ind) @ np.array(list(map(phi, [1,2,3], [x]*3)))
        #   evaluates u_h at x
        if norm == 'l1':
            def integrand(x):
                return abs(
                    u(B@x + d)
                    - u_h.take(ind) @ np.array(list(map(phi, [1,2,3], [x]*3)))
                )
        elif norm == 'l2':
            def integrand(x):
                return (
                    u(B@x + d)
                    - u_h.take(ind) @ np.array(list(map(phi, [1,2,3], [x]*3)))
                )**2

        fragments[i] = integrator.integrate(integrand) * detj

    # this is more accurate than using len(faces) += operations
    s = fragments.sum()

    if norm == 'l1':
        return s
    elif norm == 'l2':
        return s**0.5
