import numpy as np


class Cell():
    def __init__(self, vertices, topology, lower_element):
        self._vertices = np.array(vertices, dtype=np.double)
        self._topology = topology
        self._lower_element = lower_element
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

    @property
    def lower_element(self):
        return self._lower_element


class Interval(Cell):
    def __init__(self, vertices, topology):
        Cell.__init__(self, vertices, topology, None)


    def affine_transform(self, e):
        return lambda x: (e[1]-e[0]) * x + e[0]

    def affine_transform_jacobian(self, e):
        jt = e[1] - e[0]

        if len(jt) == 1:
            detj = jt
        else:
            detj = np.dot(jt,jt)**0.5

        return np.array([[detj]]), np.array([[1/detj]])


class Triangle(Cell):
    def __init__(self, vertices, topology, lower_element):
        Cell.__init__(self, vertices, topology, lower_element)


    def affine_transform(self, e):
        B = np.array([e[1] - e[0], e[2] - e[0]], dtype=np.double).T
        d = e[0]

        return lambda x: B@x + d


    def affine_transform_jacobian(self, e):
        '''
        Jacobian and Mapping needed to compute integrals of gradients of basis
        functions.
        Assumes the mapping F: T -> e, where T is this cell (self) and :param e:

        Computes |JF| and D(F^-1) (which agrees with (DF)^-1

        Straig forward computation if spatial dimensions of T and e agree.
        In the case of an embedded manifold, gernalized jacobian and
        left-inverse are computed as F has to be injective.
        '''
        jt = np.array([e[1] - e[0], e[2] - e[0]], dtype=np.double)

        if jt.shape[-1] == jt.shape[-2]:
            detj = np.abs(np.linalg.det(jt))
            jTinv = np.linalg.inv(jt)
        else:
            detj = np.linalg.det(jt@jt.T)**0.5
            jTinv = np.linalg.pinv(jt)
            jTinv = np.round(jTinv)

        return detj, jTinv


class Tetrahedron(Cell):
    def __init__(self, vertices, topology, lower_element):
        raise NotImplementedError()
        Cell.__init__(self, vertices, topology, lower_element)

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
                                  2: [2,0]}},
                             referenceInterval)
