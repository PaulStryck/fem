import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation

from fem.reference_elements import referenceTriangle
from fem.finite_elements import PElement
from fem.function_space import FEFunctionSpace


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

