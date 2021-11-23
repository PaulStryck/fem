import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from fem.fefunction import FEFunction
from fem.finite_elements import PElement
from fem.function_space import FEFunctionSpace
from fem.mesh import SimplexMesh
from fem.reference_elements import referenceInterval, referenceTriangle


def int_to_sub(s: str):
    if int(s) < 0: raise ValueError()
    if len(s) == 1: return chr(8320 + int(s))
    if len(s) > 1:
        return chr(8320 + int(s[0])) + int_to_sub(s[1:])


def plot_basis(fs: FEFunctionSpace):
    f = FEFunction(fs)

    fig = plt.figure(dpi=600)
    cbar_ax = fig.add_subplot(1,1,1)
    cbar_ax.axis('off')
    if fs.dim <= 3:
        subplt = (1, fs.dim)
    else:
        subplt = (int(np.floor(np.sqrt(fs.dim))),
                  int(np.ceil(np.sqrt(fs.dim))))

    projection = None if fs.mesh.dim == 1 else '3d'

    for i in range(fs.dim):
        ax = fig.add_subplot(*subplt, i+1, projection=projection)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # add mesh outline
        if fs.mesh.dim > 1:
            for e in fs.mesh.nfaces[1].masked_view:
                ax.plot(*(fs.mesh.nfaces[0].arr[e].T), 'k')

        # plot ith basis function
        cffs = np.zeros(fs.dim)
        cffs[i] = 1
        f.coefficients = cffs
        f.plot(ax, deg=20)

        ax.set_title(u'\u03a6{}'.format(int_to_sub(str(i))))

    # ax.view_init(30, -60)

    fig.colorbar(ScalarMappable(norm=Normalize(*(0,1)), cmap='plasma'),
                 ax = cbar_ax,
                 location = 'top',
                 orientation = 'horizontal')
    plt.savefig("figs/p1_mesh_basis.png", bbox_inches="tight")


if __name__ == '__main__':
    m = SimplexMesh(
        referenceTriangle.vertices,
        np.array([[0, 1, 2]]),
        referenceTriangle
    )
    # m = SimplexMesh(
    #     referenceInterval.vertices,
    #     np.array([[0, 1]]),
    #     referenceInterval
    # )

    m = SimplexMesh.Create_2d_unit_square_structured(3)

    fs = FEFunctionSpace(m, PElement(1, m.element))
    plot_basis(fs)

