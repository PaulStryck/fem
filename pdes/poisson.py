import numpy as np
from scipy.sparse import bmat, coo_matrix
from scipy.sparse.linalg import spsolve

from fem.finite_elements import PElement
from fem.mesh import SimplexMesh
from fem.function_space import FEFunctionSpace, FEFunction
from fem.integration import gauss_legendre_quadrature


def asm(fs):
    n = 0
    numDataPts = fs.mesh.entities_per_dimension[-1] * fs.element.dim**2
    _i = np.empty(numDataPts, dtype=int)
    _j = np.empty(numDataPts, dtype=int)
    _data = np.empty(numDataPts, dtype=np.double)

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
        G = np.empty(dPhi.shape)
        for i in range(dPhi.shape[0]):
            for j in range(dPhi.shape[1]):
                G[i,j] = jTinv @ dPhi[i,j]

        # compute local stiffness
        localStiffness = np.zeros((dPhi.shape[1],dPhi.shape[1]))
        for i,j in np.ndindex(localStiffness.shape):
            for l in range(dPhi.shape[0]):
                localStiffness[i,j] += (np.dot(G[l,i], G[l,j]) * quadrature.weights[l])

            localStiffness[i,j] *= detj

        # compute local mass
        localMass = Phi.T @ (np.diag(quadrature.weights) @ Phi)
        localMass *= detj

        # map localStiffness to global stifness
        for k,l in np.ndindex(localStiffness.shape):
            _i[n] = ind[k]
            _j[n] = ind[l]

            _data[n] = localStiffness[k,l]
            _data_m[n] = localMass[k,l]
            n += 1

        # compute projection of RHS

    _f = FEFunction(fs)
    _f.interpolate(lambda x: 1)

    mass = coo_matrix((_data_m, (_i, _j)))


    return coo_matrix((_data, (_i, _j))), mass@_f.coefficients


def poisson(m, f, s):
    fe = PElement(1, m.element)
    fs = FEFunctionSpace(m, fe)

    # FEFunctionSpace should be aware of BCs?!
    # FEFunctionSpace is part of the co-chain complex, thus should exhibit a
    # boundary operator

    A, b = asm(fs)

    # fs_bound_0 = set([0,1,2,3,5,6,7,8])
    fs_bound_0 = set([3,5,6,7,8])
    # fs_bound_1 = set([0,2,5,12,11,4,9,15])
    # fs_bound_1 = set([5,12,11,4,9,15])

    bound = []

    for e in fs_bound_0:
        g = fs.glob(0, [e])
        for n in np.arange(g, g+1):
            bound.append(n)

    # for e in fs_bound_1:
    #     g = fs.glob(1, [e])
    #     for n in np.arange(g, g+1):
    #         bound.append(n)

    # TODO: non-zero Neumann BC must be taken into account here

    # assembling the projection matrix for dirichlet boundaries
    #fs_bound = fs.boundaries()  # subspace of the entire FEFunctionSpace
                                # representing only the boundary
                                # Must be of the same elementType just one Dim
                                # lower
    # boundary_function = fs_bound.project(lambda x: 0)
    l = len(bound)
    n = fs.dim
    R = coo_matrix(
        (np.ones(l), (np.array(bound), np.arange(l))),
        shape=(n, l)
    )

    # assemble complete system matrix with enforced dirichlet conditions on
    # boundary with R
    # [[A   R]
    #  [R^t 0]]
    sys = bmat([[A, R], [R.transpose(), None]])

    # assemble rhs of complete system. First n*n entries are the
    # actual right hand side. It follows the enforced dirichlet condition
    rhs = np.concatenate((b, np.zeros(len(bound))))
    x = spsolve(sys.tocsr(), rhs)
    _f = FEFunction(fs)
    _f._coefficients = x[:fs.dim]
    _f.plot()


if __name__ == '__main__':
    np.set_printoptions(precision=2, linewidth=178)

    k = l = 2
    # rhs
    def f(x):
        return ((l*math.pi)**2 + (k*math.pi)**2) * math.sin(l*x[0]*math.pi)*math.sin(k*x[1]*math.pi)

    # analytical solution
    def s(x):
        return math.sin(l*x[0]*math.pi)*math.sin(k*x[1]*math.pi)


    m = SimplexMesh.Create_2d_unit_square_structured(3)

    poisson(m, f, s)
