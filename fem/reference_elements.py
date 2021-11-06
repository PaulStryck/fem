import numpy as np


class Cell():
    def __init__(self, vertices, topology):
        self._vertices = np.array(vertices, dtype=np.double)
        self._topology = topology
        self._dim = self._vertices.shape[1]


    def affine_transform(self, element):
        raise NotImplementedError()


    def affine_transform_jacobian(self, element):
        raise NotImplementedError()


    @property
    def vertices(self):
        return self._vertices


    @property
    def topology(self):
        return self._topology


    @property
    def dim(self):
        return self._dim


class Interval(Cell):
    def __init__(self, vertices, topology):
        Cell.__init__(self, vertices, topology)


    def affine_transform(self, e):
        return lambda x: (e[1]-e[0]) * x + e[0]

    def affine_transform_jacobian(self, e):
        j = e[1] - e[0]

        return j, 1/j


class Triangle(Cell):
    def __init__(self, vertices, topology):
        Cell.__init__(self, vertices, topology)


    def affine_transform(self, e):
        B = np.array([e[1] - e[0], e[2] - e[0]], dtype=np.double)
        d = e[0]

        return lambda x: B@x + d


    def affine_transform_jacobian(self, e):
        jt = np.array([e[1] - e[0], e[2] - e[0]], dtype=np.double)

        detj = np.abs(np.linalg.det(jt))
        jTinv = np.linalg.inv(jt)

        return detj, jTinv


class Tetrahedron(Cell):
    def __init__(self, vertices, topology):
        raise NotImplementedError()
        Cell.__init__(self, vertices, topology)

'''
Topology Data structure:
Dict[dim, cell] -> [vertex]

'''
referenceInterval = Interval([[0],[1]],
                             {0: {0: [0],
                                  1: [1]},
                              1: {0: [0,1]}})

referenceTriangle = Triangle([[0,0], [1,0], [0,1]],
                             {0: {0: [0],
                                  1: [1],
                                  2: [2]},
                              1: {0: [0,1],
                                  1: [1,2],
                                  2: [2,0]}})
