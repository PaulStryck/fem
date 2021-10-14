import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.interpolate import griddata
from scipy.sparse import bmat, coo_matrix
from scipy.sparse.linalg import spsolve

import fem
import mesh


def main():
    pts = 10
    m = mesh.unit_square(pts)
    # m = mesh.import_gmsh('untitled.msh')

    # assemble a and b from Ax=b
    a, b = fem.asm_system(m)

    n, _ = a.shape


    l = len(m.d_boundaries)
    r = coo_matrix((np.ones(l), (m.d_boundaries, np.arange(l))),
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

    plot_result_unit_square(m, x[:n])



def plot_result_unit_square(m: mesh.Mesh, x):
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
