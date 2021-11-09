from itertools import combinations

import numpy as np
from scipy.spatial import Delaunay

from fem.reference_elements import referenceTriangle, referenceInterval


class SimplexMesh:
    '''
    n dimensional simplex mesh. Mesh is a k-dim submanifold of an n>=2
    dimensional space. Obviously k<=n.
    '''

    @classmethod
    def Create_2d_unit_square_unstructured(cls, n):
        # include all corner points
        corners = np.array([[0,0], [1,0], [1,1], [0,1]])

        # include sufficient points on all edges
        pts = np.random.uniform(low=[0], high=[1], size=(4, n-2))
        e_0 = np.array([pts[0], np.zeros(n-2)]).T
        e_1 = np.array([pts[1], np.zeros(n-2)]).T[:,[1, 0]]
        e_2 = np.array([pts[2], np.ones(n-2)]).T
        e_3 = np.array([pts[3], np.ones(n-2)]).T[:,[1, 0]]

        inner = np.random.uniform(low=[0,0], high=[1,1], size=(n**2-4*(n-2)-4,2))
        all_pts = np.vstack([corners, e_0, e_1, e_2, e_3, inner])

        mesh = Delaunay(all_pts)

        return cls(mesh.points, mesh.simplices, referenceTriangle)


    @classmethod
    def Create_2d_unit_square_structured(cls, n):
        if n < 2:
            n = 2

        h = 1 / (n-1)
        k = n*n            # Number of vertices
        l = 2*(n-1)*(n-1)  # Number of faces
        vertices = np.empty((k,2), dtype=np.double)
        faces    = np.empty((l,3), dtype=np.uint)

        # Build vertex list
        # bottom left to top right, row wise
        for i in range(n):
            for j in range(n):
                vertices[i*n+j] = [j * h, i * h]

        for i in range(n-1):
            for j in range(n-1):
                ind = 2 * (i*(n-1) + j)
                faces[ind]   = [i*n + j  , i*n + j + 1  , (i+1)*n + j]
                faces[ind+1] = [i*n + j+1, (i+1)*n + j+1, (i+1)*n + j]

        return cls(vertices, faces, referenceTriangle)


    @classmethod
    def Create_2d_manifold(cls, n):
        if n < 2:
            n = 2

        h = 1 / (n-1)
        k = n*n            # Number of vertices
        l = 2*(n-1)*(n-1)  # Number of faces
        vertices = np.empty((k,3), dtype=np.double)
        faces    = np.empty((l,3), dtype=np.uint)

        # Build vertex list
        # bottom left to top right, row wise
        f = lambda x,y: np.sin(-(x-0.5)*(y-.5))
        for i in range(n):
            for j in range(n):
                vertices[i*n+j] = [j * h, i * h, f(i*h,j*h)]

        for i in range(n-1):
            for j in range(n-1):
                ind = 2 * (i*(n-1) + j)
                faces[ind]   = [i*n + j  , i*n + j + 1  , (i+1)*n + j]
                faces[ind+1] = [i*n + j+1, (i+1)*n + j+1, (i+1)*n + j]

        return cls(vertices, faces, referenceTriangle)

    @classmethod
    def Create_1d_unit_interval_structured(cls, n):
        if n < 2:
            n = 2

        vertices = np.expand_dims(np.linspace(0,1,n), 1)
        faces    = np.array([[i,i+1] for i in range(n-1)])

        return cls(vertices, faces, referenceInterval)


    def __init__(self, vertices, cells, element):
        '''
        :param vertices: list of vertices in n dimensional space.
                         Must be a numpy ndarray of shape (k,n). Where k is the
                         number of vertices an n is the dimension of the space.

        :param cells: list of elements in form of a vertex list.
                         Must be a numpy ndarray of shape (l, d). Where l is
                         the number of elements and d is the number of points
                         describing each simplex.
                         Thus, d-1 is the dimensionality of the submanifold the
                         mesh describes.
                         I.e.,
                         d = 2 => mesh of lines.
                         d = 3 => mesh of triangles.

        Caution! Only works for orientable surfaces. Gives garbage results for
        möbius strips!

        Create numberings for all topological entities within their dimension.
        I.e., number all vertices from 0 to n, all edges from 0 to m, all faces
        from 0 to l, and so on.

        Also create lookup tables for adjacent, lower dimensional entites in
        all dimensions.
        I.e., Which vertices are the edges made of.
              Which edges are the faces made of,
              Which vertices are the faces made of

        For a 2D simplex Mesh, a global numbering for all vertices, edges and
        faces is needed.
        vertices are numbered implicitly by the order in :param vertices:
        faces are numbered implicitly by the order in :param elements:
        An edge numbering must be created

        For a 1D simplex mesh, the entire numbering is implicitly given.
        Only 0D and 1D numbering is needed. This is contained in :param
        vertices: and :param elements:
        '''

        # dimension of vertex coordinate vectores
        self.dim             = vertices.shape[1]

        # dimension of the mesh
        self.dim_submanifold = element.dim

        # if self.dim != 2 and self.dim_submanifold != 2:
        #     raise NotImplementedError()

        if self.dim_submanifold > self.dim:
            raise ValueError(
                'Cannot embed a {} dimensional manifold into a {} dimensional'
                + ' space'.format(self.dim_submanifold, self.dim)
            );

        if element.dim != self.dim_submanifold:
            raise ValueError("Mesh Element does not match manifold dimension")

        self._element = element

        # self.nfaces doubles as the global numbering for all entities within
        # their dimension.
        # And vertex list for dimension n to dimension 0, for n > 0
        # And actual spatial coordinates for n = 0
        # Everything in self.nfaces has a globally implied direction.
        #   n = 0: (Implied to be of [x_1, x_2, ..., x_n] form
        #   n = 1: Lower vertex id to higher vertex id
        #   n = 2: Counterclockwise
        self.nfaces = {
            0: vertices
        }

        for n in range(1, self.dim_submanifold):
            # create global edge numbering, where the global direction is always
            # low to high
            _edges = np.array(list(set(tuple(sorted(e))
                                       for t in cells
                                       for e in combinations(t, 2))))
            self.nfaces[n] = _edges

        self.nfaces[self.dim_submanifold] = cells

        self.entity_numbering = dict([(i, list(range(len(self.nfaces[i]))))
                                      for i in self.nfaces.keys()])

        self._entities_per_dimension = np.array(
            [self.nfaces[n].shape[0] for n in sorted(self.nfaces)]
        )

        if self.dim_submanifold == 2:
            # self.klookup[i][j] is a mapping from j-entities to i-entites
            # where j > i, and i > 0. For i = 0 refer to self.nfaces
            # self.klookup[i][j] -> [[({-1,1}, a)]*b]*c
            #   where {-1,1} is the local direction relative to the global one
            #         a is the id of the respective global entity ID
            #         b is how many i entities each j-entity consists of
            #         c is how many j-entities the mesh is made of
            self._klookup = {
                1: {2: None}
            }


            # create inverse function of self.nfaces[1]
            # _edge_lookup: Edge -> (Direction, GlobalEdgeID)
            # Where Edge \in (VertexID, VertexID)
            _edge_lookup = {tuple(e): (d, i)
                            for i, e_ in enumerate(self.nfaces[1])
                            for d, e in ((1, e_), (-1, reversed(e_)))}

            self._klookup[1][2] = [[_edge_lookup[(e[i], e[(i+1)%3])]
                                   for i in range(3)]
                                  for e in self.nfaces[2]]

        # The boundary is the list of all (self.dim_submanifold - 1)-entities
        # that are adjacent to exactly one (self.dim_submanifold)-entity
        _adjacency_count = np.repeat(
            2,
            self.nfaces[self.dim_submanifold-1].shape[0]
        )

        d_sub = self.dim_submanifold
        for es in self.adjacency(d_sub,d_sub-1):
            for e in es:
                if type(e) is tuple:
                    _, e = e
                _adjacency_count[e] -= 1

        self._boundary_cells = np.where(_adjacency_count == 1)[0]
        self._boundary_mesh = None

        if len(self.boundary_cells) > 0:
            d = self.dim_submanifold
            boundary_cells = self.adjacency(d-1,0)[self.boundary_cells]

            self._boundary_mesh = SubSimplexMesh(outer=self,
                                                 cells=boundary_cells,
                                                 n_lookup=self.boundary_cells)


    @property
    def element(self):
        return self._element


    @property
    def boundary_cells(self):
        return self._boundary_cells

    @property
    def boundary_mesh(self):
        return self._boundary_mesh


    @property
    def entities_per_dimension(self):
        return self._entities_per_dimension


    def adjacency(self, d1, d2):
        '''
        Get d2-entities adjacent to d1-entities

        Only implemented properly for d2 < d1
        For d2==d1: Each element is only ajacent to itself

        For d2=1, wrapper for self.nfaces
        Otherwise wrapper for self.klookup
        '''
        if d1 < 0 or d2 < 0:
            raise ValueError("dimensions must be positive")

        if d1 > self.dim_submanifold:
            raise ValueError("d1 must be less or equal to self.dim_submanifold")

        if d2 > d1:
            raise NotImplementedError()

        if d1 == d2:
            l = len(self.nfaces[d1])
            return np.arange(l).reshape(l,-1)

        # from here d2 < d1, both positive and d1 meaningful

        if d2 == 0:
            return self.nfaces[d1]

        return self._klookup[d2][d1]


    # TODO: figure out how to implement boundary operator. This should return a
    # list of all connected boundaaies of the submanifold.

class SubSimplexMesh(SimplexMesh):
    def __init__(self, outer: SimplexMesh, cells, n_lookup):
        SimplexMesh.__init__(self,
                             vertices=outer.nfaces[0],
                             cells=cells,
                             element=outer.element.lower_element)

        # TODO: Build lookup table for outer numbering
        # Vertex numbering stays the same
        # Highest dimensional numbering table is known
        self.entity_numbering[self.dim_submanifold] = n_lookup
