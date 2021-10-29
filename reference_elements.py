import numpy as np

class ReferenceElement():
    def __init__(self, vertices):
        self._vertices = np.array(vertices, dtype=np.double)

        self._dim = self._vertices.ndim


    @property
    def vertices(self):
        return self._vertices


    @property
    def dim(self):
        return self._dim


referenceInterval = ReferenceElement(
    vertices = [0, 1]
)

referenceTriangle = ReferenceElement(
    vertices = [[0,0], [0,1], [1,0]]
)

referenceTetrahedron = ReferenceElement(
    vertices = [[0,0,0], [1,0,0], [0,1,0], [0,0,1]]
)
