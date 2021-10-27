import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.interpolate import griddata
from scipy.sparse import bmat, coo_matrix
from scipy.sparse.linalg import spsolve

import error
import fem
import mesh
from mesh import Mesh, MeshFunction


def poisson(m: Mesh, dirichlet_fn, neumann_fn, f) -> MeshFunction:
    # assemble a from Ax=b
    print("Assemble Laplacian")
    A = fem.asm_system(m)

    # assemble mass matrix M_{i,j} = \int_{\Omega}\phi_i*\phi_j\dx
    print("Assemble Mass")
    M = fem.asm_mass(m)

    # rhs of Ax=b
    print("Assemble RHS")
    b = M @ np.array([f(x) for x in m.vertices])

    n, _ = A.shape

    print("Introducing BCs")
    l = len(m.d_boundaries)
    d_b_vertices = map(dirichlet_fn,
                       [m.vertices[i] for i in m.d_boundaries])

    n_b_vertices = map(neumann_fn,
                       [m.vertices[i] for i in m.n_boundaries])

    # assembling the projection matrix for dirichlet boundaries
    r = coo_matrix((np.ones(l), (m.d_boundaries, np.arange(l))),
                   shape=(n,l) )

    # assemble complete system matrix with enforced dirichlet conditions on
    # boundary with r
    # [[A   R]
    #  [R^t 0]]
    sys = bmat([[A, r], [r.transpose(), None]])

    # assemble rhs of complete system. First n*n entries are the
    # actual right hand side. It follows the enforced dirichlet condition
    rhs = np.concatenate((b, np.array(list(d_b_vertices))))

    print('Solving a system of dim {}'.format(sys.shape))

    # x[:n] contains solution vector
    # x[n:] contains lagrange multiplies. can be discarded
    x = spsolve(sys.tocsr(), rhs)

    return (x[:n], m)


def main():
    dirichlet_fn = lambda x: 0  # zero dirichlet boundary

    # TODO: neumann condition not yet applied to the solution
    neumann_fn   = lambda x: 0  # zero neumann boundary

    # eigenfunction source term
    k = l = 2
    def f(x):
        return ((l*math.pi)**2 + (k*math.pi)**2) * math.sin(l*x[0]*math.pi)*math.sin(k*x[1]*math.pi)

    # analytical solution
    def s(x):
        return math.sin(l*x[0]*math.pi)*math.sin(k*x[1]*math.pi)

    # store l1 and l2 errors ||u - uh||
    err_l1 = []
    err_l2 = []

    for i in range(2,30):
        print("Assemble Grid")
        pts = i
        m = mesh.unit_square(pts)
        # m = mesh.import_gmsh('untitled.msh')

        uh = poisson(m, dirichlet_fn, neumann_fn, f)

        print("Compute Errors")
        err_l1.append( error.error_norm(s, uh, 'l1') )
        err_l2.append( error.error_norm(s, uh, 'l2') )

    plt.plot(err_l1, '+')
    plt.plot(err_l2, '.')
    plt.show()

    # plot_result_unit_square(uh)


def plot_result_unit_square(uh):
    x, m = uh

    # how many points to interpolate for visualization
    nx, ny = 1000, 1000

    # all vertices of the used grid
    points = np.array(m.vertices)

    # interpolate result for visualization
    grid_x, grid_y = np.meshgrid(np.linspace(0,1,nx), np.linspace(0,1,ny))
    grid_z1 = griddata((points[:,0], points[:,1]), x,
                       (grid_x, grid_y), method='linear')

    # Setup the figure with 3d plot on the left and 2d heatmap to the right
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    # 3d surface
    pos = ax1.plot_surface(grid_x, grid_y, grid_z1,
                           cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # 2d heatmap
    pos = ax2.imshow(grid_z1.T, extent=[0, 1, 0, 1], cmap=cm.coolwarm)

    # overlay the grid
    ax2.triplot(points[:,0], points[:,1], m.faces, linewidth=0.2)

    fig.colorbar(pos, ax=ax2)

    plt.show()


if __name__ == "__main__":
    np.set_printoptions(precision=2, linewidth=178)
    main()
