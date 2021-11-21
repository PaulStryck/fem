import matplotlib.pyplot as plt
import numpy as np

from fem.fefunction import FEFunction
from fem.finite_elements import PElement
from fem.function_space import FEFunctionSpace
from fem.mesh import SimplexMesh
from fem.reference_elements import referenceTriangle, referenceInterval

def plot_basis(fs: FEFunctionSpace):
    f = FEFunction(fs)

    fig = plt.figure()
    if fs.dim <= 3:
        subplt = (1, fs.dim)
    else:
        subplt = (int(np.floor(np.sqrt(fs.dim))),
                  int(np.ceil(np.sqrt(fs.dim))))

    projection = None if fs.mesh.dim == 1 else '3d'

    for i in range(fs.dim):
        ax = fig.add_subplot(*subplt, i+1, projection=projection)

        # add mesh outline
        if fs.mesh.dim > 1:
            for e in m.nfaces[1].masked_view:
                ax.plot(*(m.nfaces[0].arr[e].T), 'k')

        # plot ith basis function
        cffs = np.zeros(fs.dim)
        cffs[i] = 1
        f.coefficients = cffs
        f.plot(ax, deg=20)

    plt.show()


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

    # m = SimplexMesh.Create_2d_unit_square_structured(2)

    fs = FEFunctionSpace(m, PElement(1, m.element))

    plot_basis(fs)

