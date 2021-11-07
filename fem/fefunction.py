import numpy as np
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import Normalize

from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D

from fem.finite_elements import PElement
from fem.function_space import FEFunctionSpace
from fem.reference_elements import referenceTriangle, referenceInterval


class FEFunction():
    def __init__(self, fs):
        self._fs = fs
        self._coefficients = np.array([np.NAN]*fs.dim)


    def interpolate(self, f):
        for i, (pts, globi) in enumerate(self._fs.elements):
            f_i = self._fs.element.cell.affine_transform(pts)
            for k, l in enumerate(globi):
                if self._coefficients[l] == np.NAN:
                    continue
                node = f_i(self._fs.element.nodes[k])
                self._coefficients[l] = f(node)


    @property
    def coefficients(self):
        return self._coefficients

    @property
    def fs(self):
        return self._fs


    def plot_submanifold(self):
        fs = self._fs

        d = 2 * (fs.element.deg + 1)

        if fs.element.cell is referenceInterval:
            local_coords = np.expand_dims(np.linspace(0, 1, d), 1)
        elif fs.element.cell is referenceTriangle:
            local_coords, triangles = self._lagrange_triangles(d)
        else:
            raise ValueError("Unknown reference cell: %s" % fs.element.cell)

        n_bpts = len(local_coords)

        f_eval = fs.element.phi_eval(local_coords)

        c_e = PElement(1, fs._mesh.element)
        c_fs = FEFunctionSpace(fs._mesh, c_e)
        c_eval = c_fs.element.phi_eval(local_coords)

        xs = np.empty((fs._mesh.dim, n_bpts *
                       fs._mesh.entities_per_dimension[-1]))

        vs = np.empty(n_bpts * fs._mesh.entities_per_dimension[-1])
        ts = np.empty((len(triangles) * fs._mesh.entities_per_dimension[-1],3))
        values = np.empty(len(triangles) * fs._mesh.entities_per_dimension[-1])
        n_ts = len(triangles)

        for c in range(fs._mesh.entities_per_dimension[-1]):
            vertex_coords = fs._mesh.nfaces[0][c_fs.mapping[c, :], :]
            x = np.dot(c_eval, vertex_coords)

            local_function_coefs = self.coefficients[fs.mapping[c,:]]
            val = np.dot(f_eval, local_function_coefs)

            xs[:,c*n_bpts:(c+1)*n_bpts] = x.T
            ts[c*n_ts:(c+1)*n_ts] = triangles + c*n_bpts
            for i in range(len(triangles)):
                values[c*n_ts+i] = val[triangles[i].astype(int)].mean()

        ax = plt.figure().add_subplot(projection='3d')
        p3dc = ax.plot_trisurf(Triangulation(xs[0], xs[1], ts),
                               xs[2], color='r', linewidth=0)

        x, y, z, _ = p3dc._vec
        slices = p3dc._segslices
        triangles = np.array([np.array((x[s],y[s],z[s])).T for s in slices])

        xb, yb, zb = triangles.mean(axis=1).T

        # usual stuff
        norm = Normalize()
        colors = get_cmap('summer')(norm(values))

        # set the face colors of the Poly3DCollection
        p3dc.set_fc(colors)

        plt.show()


