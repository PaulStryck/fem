from itertools import combinations
from os import wait
from typing import Callable, Optional, Union

import numpy as np
import numpy.typing as npt
from scipy.spatial import Delaunay

from fem.reference_elements import Cell, referenceInterval, referenceTriangle


class MaskedList():
    def __init__(self,
                 arr: npt.NDArray,
                 mask: Optional[npt.NDArray[np.bool_]] = None):
        if mask is None:
            self._mask = np.array([False]*arr.shape[0], dtype=np.bool_)
        else:
            self._mask = mask

        self._arr = arr

        if self.arr.shape[0] != self.mask.shape[0]:
            raise ValueError("Mask length must match arr length")

    @property
    def masked_view(self):
        return self._arr[~self.mask].view()

    @property
    def arr(self):
        return self._arr.view()

    @property
    def mask(self):
        return self._mask.view()

    @mask.setter
    def mask(self, mask):
        if self.arr.shape[0] != mask.shape[0]:
            raise ValueError("Mask length must match arr length")
        self._mask = mask

    def halfdeepcopy(self, mask: Optional[npt.NDArray[np.bool_]]):
        if mask is None:
            return MaskedList(self._arr.view(), self._mask.copy())

        return MaskedList(self._arr.view(), mask)



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


    def __init__(self,
                 _v: Union[MaskedList, npt.ArrayLike],
                 _c: Union[MaskedList, npt.ArrayLike],
                 element : Cell):
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
        vertices = _v if type(_v) is MaskedList else MaskedList(np.array(_v))
        cells    = _c if type(_c) is MaskedList else MaskedList(np.array(_c))

        # dimension of vertex coordinate vectores
        self.dim             = vertices.arr.shape[1]

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
            _edges = list(set(tuple(sorted(e))
                              for t in cells.masked_view
                              for e in combinations(t, 2)))
            self.nfaces[n] = MaskedList(np.array(_edges))

        self.nfaces[self.dim_submanifold] = cells

        self._entities_per_dimension = np.array(
            [self.nfaces[n].masked_view.shape[0] for n in sorted(self.nfaces)]
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
                            for i, e_ in enumerate(self.nfaces[1].arr)
                            for d, e in ((1, e_), (-1, reversed(e_)))}

            self._klookup[1][2] = [[_edge_lookup[(e[i], e[(i+1)%3])]
                                   for i in range(3)]
                                  for e in self.nfaces[2].masked_view]

        # The boundary is the list of all (self.dim_submanifold - 1)-entities
        # that are adjacent to exactly one (self.dim_submanifold)-entity
        _adjacency_count = np.repeat(
            2,
            self.nfaces[self.dim_submanifold-1].arr.shape[0]
        )

        d_sub = self.dim_submanifold
        for es in self.adjacency(d_sub,d_sub-1):
            for e in es:
                if type(e) is tuple:
                    _, e = e

                _adjacency_count[e] -= 1

        _adjacency_count[self.nfaces[d_sub-1].mask.nonzero()[0]] = 2

        self._interior_facets = np.where(_adjacency_count == 0)[0]
        self._boundary_mesh = None

        n_facets = self.nfaces[d_sub-1].masked_view.shape[0]
        n_interior_facets = self._interior_facets.shape[0]

        if (n_facets - n_interior_facets) > 0 and d_sub > 1:
            masked_cells = MaskedList(self.nfaces[d_sub-1].arr.view())
            masked_cells.mask[self._interior_facets] = True

            self._boundary_mesh = SubSimplexMesh(outer=self,
                                                 cells=masked_cells)


    @property
    def element(self):
        return self._element

    @property
    def boundary_mesh(self):
        return self._boundary_mesh


    @property
    def entities_per_dimension(self):
        return self._entities_per_dimension

    @property
    def global_entities_per_dimension(self):
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
            l = len(self.nfaces[d1].arr)
            mask = self.nfaces[d1].mask
            return np.arange(l)[~mask].reshape(-1,1)
            # return np.arange(l).reshape(-1,1)

        # from here d2 < d1, both positive and d1 meaningful

        if d2 == 0:
            return self.nfaces[d1].masked_view

        return self._klookup[d2][d1]

    def split(self, predicate: Callable[[npt.NDArray],bool]):
        d_sub = self.dim_submanifold

        cond = np.array(
            [all(map(predicate, self.nfaces[0].arr[cells]))
             for cells in self.nfaces[d_sub].arr]
        )
        ma_1 = np.logical_or(self.nfaces[d_sub].mask, ~cond)
        ma_2 = np.logical_or(self.nfaces[d_sub].mask, cond)

        m_1 = SubSimplexMesh(outer=self,
                             cells=self.nfaces[d_sub].halfdeepcopy(ma_1))
        m_2 = SubSimplexMesh(outer=self,
                             cells=self.nfaces[d_sub].halfdeepcopy(ma_2))
        return m_1, m_2


    # TODO: figure out how to implement boundary operator. This should return a
    # list of all connected boundaaies of the submanifold.

class SubSimplexMesh(SimplexMesh):
    def __init__(self, outer: SimplexMesh, cells: MaskedList):
        # all vertices initially masked
        vertices = MaskedList(outer.nfaces[0].arr,
                              np.array([True]*outer.nfaces[0].arr.shape[0]))

        # unmask all needed vertices
        for c in cells.masked_view:
            vertices.mask[c] = False

        d_sub = cells.arr[0].shape[0] - 1
        if outer.dim_submanifold == d_sub:
            element = outer.element
        elif outer.dim_submanifold == d_sub +1:
            element = outer.element.lower_element
        else:
            raise NotImplementedError()

        SimplexMesh.__init__(self,
                             _v=vertices,
                             _c=cells,
                             element=element)
        self._outer = outer

        # TODO: correct edge numbering
        # if d_sub == 2:
        #     self.nfaces[1] = outer.nfaces[1]
        #     self.nfaces[1].mask = np.array([True]*len(self.nfaces[1].mask))
        #     # create inverse function of self.nfaces[1]
        #     # _edge_lookup: Edge -> (Direction, GlobalEdgeID)
        #     # Where Edge \in (VertexID, VertexID)
        #     _edge_lookup = {tuple(e): (d, i)
        #                     for i, e_ in enumerate(self.nfaces[1].arr)
        #                     for d, e in ((1, e_), (-1, reversed(e_)))}

        #     for e in self.nfaces[2].masked_view:
        #         for i in range(3):
        #             _, n = _edge_lookup[(e[i], e[(i+1)%3])]
        #             self.nfaces[1].mask[n] = False

        #     self._klookup[1][2] = [[_edge_lookup[(e[i], e[(i+1)%3])]
        #                            for i in range(3)]
        #                           for e in self.nfaces[2].masked_view]
        #     print(self.nfaces[1].masked_view)

    @property
    def global_entities_per_dimension(self):
        return self._outer.entities_per_dimension

    def split(self, predicate: Callable[[npt.NDArray],bool]):
        d_sub = self.dim_submanifold

        cond = np.array(
            [all(map(predicate, self.nfaces[0].arr[cells]))
             for cells in self.nfaces[d_sub].arr]
        )
        ma_1 = np.logical_or(self.nfaces[d_sub].mask, ~cond)
        ma_2 = np.logical_or(self.nfaces[d_sub].mask, cond)

        m_1 = SubSimplexMesh(outer=self._outer,
                             cells=self.nfaces[d_sub].halfdeepcopy(ma_1))
        m_2 = SubSimplexMesh(outer=self._outer,
                             cells=self.nfaces[d_sub].halfdeepcopy(ma_2))
        return m_1, m_2


def import_gmsh(file: str):
    tags = [dict(), dict(), dict(), dict()]

    f = open(file, 'r')
    it = iter(f.readlines())

    while(next(it).strip() != '$MeshFormat'): pass
    v = next(it).strip().split(' ')

    # only v4.1 supported
    assert float(v[0]) == 4.1


    while(next(it).strip() != '$Entities'): pass
    t = next(it).strip().split(' ')
    numTags = [int(t[0]), int(t[1]), int(t[2]), int(t[3])]


    # all point tags
    for i in range(numTags[0]):
        l = next(it).strip().split(' ')
        if(int(l[3]) != 0):
            tags[0][int(l[0])] = int(v[4])

    # all multi dimensional tags
    for i in [1,2,3]:
        for j in range(numTags[i]):
            l = next(it).strip().split(' ')
            if(int(l[7]) != 0):
               tags[i][int(l[0])] = int(l[8])

    # skip to nodes
    while(next(it).strip() != '$Nodes'): pass

    l = next(it).strip().split(' ')
    blocks = int(l[0])
    nnodes = int(l[1])

    nodes = []
    nodeNumbering = dict()

    n = 0

    for i in range(blocks):
        l = next(it).strip().split(' ')
        nodesInBlock = int(l[3])

        # The node numbers
        for j in range(nodesInBlock):
            l = next(it).strip().split(' ')
            nodeNumbering[int(l[0])] = n
            n += 1

        # The actual coordinates
        for j in range(nodesInBlock):
            l = next(it).strip().split(' ')
            nodes.append((float(l[0]), float(l[1])))

    # skip to elements
    while(next(it).strip() != '$Elements'): pass

    l = next(it).strip().split(' ')
    blocks = int(l[0])

    edges = set()
    triangles = set()

    for i in range(blocks):
        l = next(it).strip().split(' ')
        elemDim = int(l[0])
        elemEntity = int(l[1])
        elemType = int(l[2])
        elemsInBlock = int(l[3])

        for j in range(elemsInBlock):
            l = next(it).strip().split(' ')

            if elemType == 1:
                edges.add( (0, int(l[1]), int(l[2])) )
                # edges.add( (tags[elemDim][elemEntity], int(l[1]), int(l[2])) )
            elif elemType == 2:
                triangles.add((0, int(l[1]),
                               int(l[2]), int(l[3])))
                # triangles.add((tags[elemDim][elemEntity], int(l[1]),
                #                int(l[2]), int(l[3])))
            else:
                print("Unsupported Element Type: {}".format(elemType))
                continue

    renumbering = dict()
    vertices = []
    faces = []
    d_boundaries = []
    n_boundaries = []

    for t in triangles:
        _t = (nodeNumbering[t[1]], nodeNumbering[t[2]],nodeNumbering[t[3]])

        for k in _t:
            if k not in renumbering:
                vertices.append(nodes[k])
                renumbering[k] = len(vertices)-1

        faces.append((renumbering[_t[0]], renumbering[_t[1]], renumbering[_t[2]]))

    for e in edges:
        d_boundaries.append(renumbering[nodeNumbering[e[1]]])
        d_boundaries.append(renumbering[nodeNumbering[e[2]]])

    return SimplexMesh(vertices, faces, referenceTriangle)
