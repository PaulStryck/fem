import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import Normalize
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import bmat, coo_matrix

from fem.finite_elements import PElement
from fem.function_space import FEFunctionSpace
from fem.reference_elements import referenceInterval, referenceTriangle


class FEFunction():
    def __init__(self, fs):
        self._fs = fs

        self._embedded_coefficients = None
        self._coefficients = None


    def interpolate(self, f):
        _c = []
        for pts, globi in self._fs.elements:
            f_i = self._fs.element.cell.affine_transform(pts)
            for k, l in enumerate(globi):
                node = f_i(self._fs.element.nodes[k])
                _c.append((l, f(node)))

        _c.sort()
        self._embedded_coefficients = dict(_c)


    @property
    def coefficients(self):
        return np.array(list(self._embedded_coefficients.values()))


    @coefficients.setter
    def coefficients(self, cs):
        self._embedded_coefficients = dict(enumerate(cs))

    @property
    def embedded_coeffs_indices(self):
        return np.array(list(self._embedded_coefficients.keys()))

    @property
    def embedded_coeffs_values(self):
        return np.array(list(self._embedded_coefficients.values()))


    @property
    def fs(self):
        return self._fs


    def plot_submanifold(self):
        fs = self._fs

        d = 2 * (fs.element.deg + 1)  # Render each element as d cells

        if fs.element.cell is referenceInterval:
            local_coords = np.expand_dims(np.linspace(0, 1, d), 1)
            cells = np.array([[i, i+1] for i in range(d-1)], dtype=np.uint)
        elif fs.element.cell is referenceTriangle:
            local_coords, cells = self._lagrange_triangles(d)
        else:
            raise ValueError("Unknown reference cell: %s" % fs.element.cell)

        n_bpts = len(local_coords)

        f_eval = fs.element.phi_eval(local_coords)

        c_e = PElement(1, fs._mesh.element)
        c_fs = FEFunctionSpace(fs._mesh, c_e)
        c_eval = c_fs.element.phi_eval(local_coords)

        xs = np.empty((fs._mesh.dim, n_bpts *
                       fs._mesh.entities_per_dimension[-1]))

        ts = np.empty((len(cells) * fs._mesh.entities_per_dimension[-1],
                       len(cells[0])), dtype=np.uint)
        values = np.empty(len(cells) * fs._mesh.entities_per_dimension[-1])
        n_ts = len(cells)

        for c in range(fs._mesh.entities_per_dimension[-1]):
            vertex_coords = fs._mesh.nfaces[0].arr[c_fs.mapping[c, :], :]
            x = np.dot(c_eval, vertex_coords)

            # TODO: fs.mapping contains global node numbers. Need local ones
            local_function_coefs = np.array([self._embedded_coefficients[k]
                                             for k in fs.mapping[c,:]])
            # local_function_coefs = self.coefficients[fs.mapping[c,:]]
            val = np.dot(f_eval, local_function_coefs)

            xs[:,c*n_bpts:(c+1)*n_bpts] = x.T
            ts[c*n_ts:(c+1)*n_ts] = cells + c*n_bpts
            for i in range(len(cells)):
                values[c*n_ts+i] = val[cells[i].astype(int)].mean()

        # color values for each cell
        norm = Normalize()
        colors = get_cmap('summer')(norm(values))

        if fs.mesh.dim == 2 and fs.mesh.dim_submanifold == 1:
            ax = plt.figure().add_subplot()
            for cell, color in zip(ts, colors):
                ax.plot(xs[0][cell], xs[1][cell], color=color)
        elif fs.mesh.dim == 3 and fs.mesh.dim_submanifold == 1:
            ax = plt.figure().add_subplot(projection='3d')
            for cell, color in zip(ts, colors):
                ax.plot(xs[0][cell], xs[1][cell], xs[2][cell], color=color)
        elif fs.mesh.dim == 3 and fs.mesh.dim_submanifold == 2:
            ax = plt.figure().add_subplot(projection='3d')
            p3dc = ax.plot_trisurf(Triangulation(xs[0], xs[1], ts),
                                   xs[2], color='r', linewidth=0)

            # set the face colors of the Poly3DCollection
            p3dc.set_fc(colors)

        plt.show()


        # set the face colors of the Poly3DCollection
        p3dc.set_fc(colors)

        plt.show()


