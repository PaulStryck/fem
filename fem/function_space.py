import numpy as np
from scipy.sparse.coo import coo_matrix

from fem.finite_elements import PElement
from fem.integration import gauss_legendre_quadrature
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
        # TODO: currently wrong due to SubSimplex consisting of too many
        # vertices
        self._dim = sum([self._mesh.entities_per_dimension[d]
                         * self._element.nodes_per_entity[d]
                         for d in range(1+self._mesh.dim_submanifold)])

        # Local -> Global node numbering lookup table
        # self._mapping[cell][loc] -> global node id
        # if this is a FEFunctionSpace on a SubSimplexMesh:
        #   cell: global on SubSimplexMesh
        #   global node id: global on SimplexMesh
        n = self._mesh.element.dim
        self._mapping = np.zeros((self._mesh.entities_per_dimension[n], self._element.dim),
                                 dtype=np.uint)

        # Generate the local -> global node numbering lookup table
        c = self._mesh.element.dim
        # TODO: adjust iterator
        # if this is a space on a submesh, i does not match
        for i in range(self._mesh.entities_per_dimension[c]):
            for d in self._element.local_nodes:  # d = dimension of entity
                N_d = self._element.nodes_per_entity[d]
                for e in self._element.local_nodes[d]:
                    loc = self._element.local_nodes[d][e]  # local entity number

                    # adjecent elements in global numbering
                    # global on SimplexMesh when using a SubSimbplexMesh
                    # whereas i is global on SubSimplexMesh
                    adj = self._mesh.adjacency(c, d)[i][e]
                    direction = 1
                    if type(adj) is tuple:
                        direction, adj = adj

                    g = self.__global(d, adj)
                    # g = self.__global(d, self._mesh.entity_numbering[d][adj])
                    self._mapping[i,loc] = np.arange(g, g+N_d, dtype=np.uint)[::direction]


    def glob(self, d, i):
        return self.__global(d,i)


    def __global(self, d, i):
        npe = self._element.nodes_per_entity
        epm = self._mesh.global_entities_per_dimension  # TODO: this must be outer mesh specific

        return int(np.dot(npe[:d], epm[:d]) + i*npe[d])


    @property
    def mapping(self):
        return self._mapping

    def project(self, f):
        raise NotImplementedError()


    def boundaries(self):
        raise NotImplementedError()


    def boundaryProjection(self):
        raise NotImplementedError()


    def asm_stiff_mass(self, stiff: bool = True, mass: bool = True):
        n = 0
        numDataPts = self.mesh.entities_per_dimension[-1] * self.element.dim**2
        _i = np.empty(numDataPts, dtype=int)
        _j = np.empty(numDataPts, dtype=int)

        _data_s = np.empty(numDataPts, dtype=np.double) if stiff else None
        _data_m = np.empty(numDataPts, dtype=np.double) if mass else None

        quadrature = gauss_legendre_quadrature(self.element.cell.dim,
                                               self.element.dim)

        # dPhi \in \R^(q,n,p), where:
        #       n: is the dimension of the polynomial space
        #       q: is the number of quadrature points
        #       p: is the dimension of element. (dimension of the range of
        #       \nabla phi)
        dPhi = self.element.grad_phi_eval(quadrature.points) if stiff else None
        Phi  = self.element.phi_eval(quadrature.points) if mass else None

        for e, ind in self.elements:
            # e, ind = (x_1, ..., x_n), (i(x_1), ... i(x_n))
            detj, jTinv = self.element.cell.affine_transform_jacobian(e)

            if stiff:
                # transform each gradient with jTinv
                G = dPhi @ jTinv.T

                # compute local stiffness
                localStiffness = np.tensordot(quadrature.weights,
                                              G@G.swapaxes(1,2),
                                              axes=1) * detj

            # compute local mass
            if mass:
                localMass = Phi.T @ (np.multiply(quadrature.weights, Phi.T).T) * detj

            # map localStiffness to global stifness
            for k,l in np.ndindex(self.element.dim, self.element.dim):
                _i[n] = ind[k]
                _j[n] = ind[l]

                if stiff:
                    _data_s[n] = localStiffness[k,l]

                if mass:
                    _data_m[n] = localMass[k,l]
                n += 1


        stiff_matrix = coo_matrix((_data_s, (_i, _j))).tocsr() if stiff else None
        mass_matrix  = coo_matrix((_data_m, (_i, _j))).tocsr() if mass  else None

        return stiff_matrix, mass_matrix

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
            if np.any(self.mesh.nfaces[0].mask[e]):
                raise ValueError("Acessing Illegal Point")

            yield self.mesh.nfaces[0].arr[e], self.mapping[i]


    @property
    def element(self):
        return self._element


    @property
    def dim(self):
        return self._dim

    @property
    def mesh(self):
        return self._mesh
