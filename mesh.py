from typing import List, Tuple

import gmsh

Coord = Tuple[float, float]
Face  = Tuple[Coord, Coord, Coord]


class Mesh:
    _vertices: List[Coord]
    _faces: List[Tuple[int, int, int]]
    _d_boundaries: list
    _n_boundaries: list

    def __init__(
        self,
        vertices: List[Tuple[float, float]],
        faces: List[Tuple[int, int, int]],
        d_boundaries: list,
        n_boundaries: list
    ):
        self._vertices = vertices
        self._faces = faces
        self._d_boundaries = d_boundaries
        self._n_boundaries = n_boundaries

    @property
    def faces(self) -> List[Tuple[int, int, int]]:
        return self._faces


    @property
    def vertices(self) -> List[Tuple[float, float]]:
        return self._vertices

    @property
    def d_boundaries(self) -> list:
        return self._d_boundaries

    @property
    def n_boundaries(self) -> list:
        return self._n_boundaries


    def __iter__(self):
        return MeshIter(self)


class MeshIter:
    _m: Mesh  = None
    _face_iterator = None

    def __init__(self, m: Mesh):
        self._m = m
        self._face_iterator = iter(self._m.faces)

    def __next__(self) -> Tuple[Face, Tuple[int, int, int]]:
        v1, v2, v3 = next(self._face_iterator)

        return (
            (self._m.vertices[v1],
             self._m.vertices[v2],
             self._m.vertices[v3]),
            (v1, v2, v3)
        )


def unit_square(n: int = 100):
    if n < 2:
        n = 2

    h = 1 / (n-1)
    vertices: List[Coord]             = []
    faces: List[Tuple[int, int, int]] = []
    d_boundaries: list                  = []
    n_boundaries: list                  = []


    # Build vertex list
    # bottom left to top right, row wise
    for i in range(n):
        for j in range(n):
            vertices.append((j * h, i * h))

    for i in range(n-1):
        for j in range(n-1):
            faces.append((i*n + j, i*n + j + 1, (i+1)*n + j))

            faces.append((i*n + j + 1, (i+1)*n + j + 1, (i+1)*n + j))

    for i in range(n):
        d_boundaries.append(i*n)  # left edge w\ lower w upper endpoint

    for i in range(1,n):
        d_boundaries.append(i)  # bottom edge w\o left w right end point

    for i in range(1, n):
        n_boundaries.append(i + (n-1)*n)  # upper edge w\o left, w\ right end points

    for i in range(1, n-1):
        n_boundaries.append(i*n+n-1)  # right edge w\o lower w\o upper end points


    return Mesh(vertices, faces, d_boundaries, n_boundaries)


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
            tags[0][int(l[0])] = int(a[4])

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

    return Mesh(vertices, faces, d_boundaries, n_boundaries)
