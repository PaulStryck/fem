import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import bmat, coo_matrix
from scipy.sparse.linalg import spsolve

from fem.error import error
from fem.fefunction import FEFunction
from fem.finite_elements import PElement
from fem.function_space import FEFunctionSpace
from fem.integration import gauss_legendre_quadrature
from fem.mesh import SimplexMesh


def asm(fs, f):
    '''
    Asseble system matrix of weak formulation for
    -∆u = f
    '''
    n = 0
    numDataPts = fs.mesh.entities_per_dimension[-1] * fs.element.dim**2
    _i = np.empty(numDataPts, dtype=int)
    _j = np.empty(numDataPts, dtype=int)

    _data_s = np.empty(numDataPts, dtype=np.double)
    _data_m = np.empty(numDataPts, dtype=np.double)

    quadrature = gauss_legendre_quadrature(fs.element.cell.dim, fs.element.dim)

    # dPhi \in \R^(q,n,p), where:
    #       n: is the dimension of the polynomial space
    #       q: is the number of quadrature points
    #       p: is the dimension of element. (dimension of the range of
    #       \nabla phi)
    dPhi = fs.element.grad_phi_eval(quadrature.points)
    Phi  = fs.element.phi_eval(quadrature.points)

    for e, ind in fs.elements:
        # e, ind = (x_1, ..., x_n), (i(x_1), ... i(x_n))
        detj, jTinv = fs.element.cell.affine_transform_jacobian(e)

        # transform each gradient with jTinv
        G = dPhi @ jTinv.T

        # compute local stiffness
        localStiffness = np.tensordot(quadrature.weights,
                                      G@G.swapaxes(1,2),
                                      axes=1) * detj

        # compute local mass
        localMass = Phi.T @ (np.multiply(quadrature.weights, Phi.T).T) * detj

        # map localStiffness to global stifness
        for k,l in np.ndindex(*localStiffness.shape):
            _i[n] = ind[k]
            _j[n] = ind[l]

            _data_s[n] = localStiffness[k,l]
            _data_m[n] = localMass[k,l]
            n += 1

    # compute projection of RHS
    _f = FEFunction(fs)
    _f.interpolate(f)

    mass      = coo_matrix((_data_m, (_i, _j)))
    stiffness = coo_matrix((_data_s, (_i, _j)))
    return stiffness, mass@_f.coefficients


def poisson(m: SimplexMesh, f, g, order=1):
    # Create Finite Dim function space
    fe = PElement(order, m.element)
    fs = FEFunctionSpace(m, fe)

    # Create finite dim subspace on boundary
    b_fe = PElement(order, m.boundary_mesh.element)
    b_fs = FEFunctionSpace(m.boundary_mesh, b_fe)

    # project function to boundary
    b_f = FEFunction(b_fs)
    b_f.interpolate(g)

    # FEFunctionSpace should be aware of BCs?!
    # FEFunctionSpace is part of the co-chain complex, thus should exhibit a
    # boundary operator

    A, b = asm(fs, f)

    # TODO: non-zero Neumann BC must be taken into account here
    n = fs.dim
    # l = b_fs.dim  TODO: Fix FS dimension calculation
    l = len(b_f.embedded_coeffs_indices)
    R = coo_matrix(
        (np.ones(l), (b_f.embedded_coeffs_indices, np.arange(l))),
        shape=(n, l)
    )

    # assemble complete system matrix with enforced dirichlet conditions on
    # boundary with R
    # [[A   R]
    #  [R^t 0]]
    sys = bmat([[A, R], [R.transpose(), None]], format='csr')

    # assemble rhs of complete system. First n*n entries are the
    # actual right hand side. It follows the enforced dirichlet condition
    rhs = np.concatenate((b, b_f.embedded_coeffs_values))
    x = spsolve(sys, rhs)
    _f = FEFunction(fs)
    _f.coefficients = x[:fs.dim]

    return _f


if __name__ == '__main__':
    np.set_printoptions(precision=2, linewidth=178)

    # RHS
    f = lambda x: - 2*(x[0]**2 - x[0] + (x[1] - 1)*x[1])

    # Dirichlet Boundary
    g = lambda _: 0

    # Actual solution
    s = lambda x: (1-x[0]) * x[0] * (1-x[1]) * x[1]

    # Gradient of actual solution
    gs = lambda x: np.array([(2*x[0]-1)*(x[1]-1)*x[1], (2*x[1]-1)*(x[0]-1)*x[0]])

    errors = {1: ([],([],[])),
              2: ([],([],[])),
              3: ([],([],[]))}

    for d in [1,2,3]:
        for n in range(2,31,2):
            # m = SimplexMesh.Create_2d_unit_square_unstructured(n)
            m = SimplexMesh.Create_2d_unit_square_structured(n)
            # m = SimplexMesh.Create_1d_unit_interval_structured(n)
            # m = SimplexMesh.Create_2d_manifold(n)
            u = poisson(m, f, g, d)

            e_l2 = error(s, u, lambda x,y: np.abs(x-y)**2)
            e_h1 = error(gs, u, lambda x,y: np.dot(x-y,x-y), grad=True)

            print("n: {}, L2 error = {}".format(n, e_l2**.5))
            print("n: {}, H1 error = {}".format(n, e_h1**.5))
            print()
            errors[d][0].append(n)
            errors[d][1][0].append(e_l2**.5)
            errors[d][1][1].append(e_l2**0.5 + e_h1**.5)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    colors = ['black', 'red', 'green']
    for c, (x,(y_1, y_2)) in zip(colors, errors.values()):
        ax1.loglog(x,y_1, c)
        ax2.loglog(x,y_2, c)

    # print(errors)
    plt.show()
