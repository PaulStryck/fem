import numpy as np

from fem.finite_elements import PElement
from fem.reference_elements import referenceTriangle

# TODO: Maybe a FEFunctionSubSpace with SubMesh with sparse matrix
#       to keep node numbering identical

class FEFunctionSpace():
    def __init__(self, mesh, element):
        self._mesh = mesh
        self._element = element

        # Degrees of freedom within the entire function space
        # This is not known a priori and computed whilst building the local to
        # global node numbering
        self._dim = 0

        # Local -> Global node numbering lookup table
        # self._mapping[cell][loc] -> global node id
        n = self._mesh.element.dim
        self._mapping = np.zeros((self._mesh.entities_per_dimension[n], self._element.dim),
                                 dtype=np.int)

        # Generate the local -> global node numbering lookup table
        c = self._mesh.element.dim
        for i, cell in enumerate(self._mesh.nfaces[c]):
            for d in self._element.local_nodes:
                N_d = self._element.nodes_per_entity[d]
                for e in self._element.local_nodes[d]:
                    loc = self._element.local_nodes[d][e]
                    adj = self._mesh.adjacency(c, d)[i][e]
                    direction = 1
                    if type(adj) is tuple:
                        direction, adj = adj

                    g = self.__global(d, adj)
                    self._mapping[i,loc] = np.arange(g, g+N_d, dtype=np.int)[::direction]

                    if self._dim < g+N_d:
                        self._dim = g+N_d


        # Compute list of boundary node indices
        # TODO: Create recursive version for readability
        bound_nodes = []
        bs = self.mesh.boundary
        for d in range(self._mesh.element.dim, 0, -1):
            # use d - 1
            for b in bs.flatten():
                N_d = self._element.nodes_per_entity[d-1]
                g = self.__global(d-1,b)
                bound_nodes.append(np.arange(g, g+N_d, dtype=np.int))
            if d > 1:
                bs = self.mesh.adjacency(d-1,d-2)[bs]

        self._boundary_nodes = np.unique(np.concatenate(bound_nodes))


    def glob(self, d, i):
        return self.__global(d,i)


    def __global(self, d, i):
        npe = self._element.nodes_per_entity
        epm = self._mesh.entities_per_dimension

        return int(np.dot(npe[:d], epm[:d]) + i*npe[d])

    @property
    def boundary_nodes(self):
        return self._boundary_nodes

    @property
    def mapping(self):
        return self._mapping

    def project(self, f):
        raise NotImplementedError()


    def boundaries(self):
        raise NotImplementedError()


    def boundaryProjection(self):
        raise NotImplementedError()

    @property
    def elements(self):
        '''
        returns a list of highest dimensional cells with corresponding indices.
        e, ind
        where e is a list of coordinates
        and ind is a list of integers representing the global node numbering of
        the corresponding coordinate
        len(list) == self.element.dim
        '''
        n = self._mesh.element.dim
        for i, e in enumerate(self._mesh.adjacency(n, 0)):
            yield self.mesh.nfaces[0][e], self.mapping[i]


    @property
    def element(self):
        return self._element


    @property
    def dim(self):
        return self._dim

    @property
    def mesh(self):
        return self._mesh
