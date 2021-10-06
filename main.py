import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import bmat, coo_matrix
import matplotlib.pyplot as plt

import mesh
import fem


def main():
    # pts = 100
    # m = mesh.unit_square(pts)
    m = mesh.import_gmsh('untitled.msh')

    # assemble a and b from Ax=b
    a, b = fem.asm_system(m)

    n, _ = a.shape


    l = len(m.boundaries)
    r = coo_matrix((np.ones(l), (list(m.boundaries), np.arange(l))),
                   shape=(n,l) )

    # assemble complete system matrix with enforced dirichlet conditions on
    # boundary with r
    # [[A   R]
    #  [R^t 0]]
    sys = bmat([[a, r], [r.transpose(), None]])

    # assemble right and side of complete system. First n*n entries are the
    # actual right hand side. It follows the enforced dirichlet condition
    rhs = np.concatenate((b, np.zeros(l)))

    print('Solving a system of dim {}'.format(sys.shape))

    # x[:n] contains solution vector
    # x[n:] contains lagrange multiplies. can be discarded
    x = spsolve(sys.tocsr(), rhs)


    points = np.array(m.vertices)
    plt.figure(figsize=(4, 4))
    plt.scatter(points[:, 0], points[:, 1], c=x[:n])
    plt.colorbar()
    plt.axis("image")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()


if __name__ == "__main__":
    np.set_printoptions(precision=2, linewidth=178)
    main()
