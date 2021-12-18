from itertools import combinations_with_replacement
from math import factorial

import numpy as np
from scipy.special import binom


def vandermonde(cell, deg, pts, dim):
    '''
    -> Vandermonde matrix for monomial basis of polynomials up to degree
    :param deg: and cell.dim spatial dimensions
        V \in \R^(n,d), n = number of points, d = dim
    '''
    n = 0
    V = np.ones((len(pts),dim))

    # vandermonde row in cell.dim spatial dimensions and polynomials of order
    # up to d
    # results prpagated column wise for all points
    for d in range(deg+1):
        for c in combinations_with_replacement(range(cell.dim), d):
            for i in c:
                V[:,n] *= pts[:,i]
            n += 1

    return V


def vandermonde_grad(cell, deg, pts, dim):
    '''
    -> dV \in \R^(n,d,cell.dim)
    '''
    n = 0
    dV = np.ones((len(pts), dim, cell.dim))

    for d in range(deg+1):
        for c in combinations_with_replacement(range(cell.dim), d):
            A = np.zeros(cell.dim)
            for i in c:
                A[i] += 1

            # derive in all spatial dimensions
            for i in range(cell.dim):
                d_A = A.copy()
                d_A[i] -= 1
                d_i = A[i]

                if d_i <= 0:
                    # derivative in ith direction is 0
                    dV[:,n,i] = 0
                else:
                    for k in range(len(A)):
                        dV[:,n,i] *= pts[:,k]**d_A[k]
                    dV[:,n,i] *= d_i
            n += 1

    return dV


class FiniteElement():
    def __init__(self, dim, cell):
        self._dim = dim
        self._cell = cell


    def phi_eval(self, pts):
        raise NotImplementedError()


    def grad_phi_eval(self, pts):
        raise NotImplementedError()


    @property
    def dim(self):
        return self._dim


    @property
    def cell(self):
        return self._cell

    @property
    def nodes(self):
        return self._nodes

    @property
    def local_nodes(self):
        return self._local_nodes

    @property
    def nodes_per_entity(self):
        return self._nodes_per_entity


class PElement(FiniteElement):
    '''
    :param deg: Polynomial degree
    :param cell: The reference cell on which this is defined. Must be a simplex
                 in this case. Must also be a subclass of Cell.
    '''
    def __init__(self, deg, cell):
        # polynomials of oder up to p
        # spatial dimenson d
        # _dim is dimension of vector space FEFunctionSpace
        p = deg
        d = cell.dim
        FiniteElement.__init__(self, int(binom(p+d,d)), cell)

        self._deg = deg

        if self.cell.dim == 1:
            self._nodes = np.expand_dims(np.linspace(0,1,self.deg+1), 1)
            self._local_nodes = {0: {0: [0],
                                     1: [len(self._nodes)-1]}}
            if self.deg > 1:
                self._local_nodes[1] = {0: list(range(1, self.deg))}
        elif self.cell.dim == 2 and self.deg == 1:
            self._nodes = self.__lagrange_pts(self.deg)
            self._local_nodes = {0: {0: [0],
                                     1: [2],
                                     2: [1]}}
        elif self.cell.dim == 2 and self.deg == 2:
            self._nodes = self.__lagrange_pts(self.deg)
            self._local_nodes = {0: {0: [0],
                                     1: [5],
                                     2: [2]},
                                 1: {0: [3],
                                     1: [4],
                                     2: [1]}}
        elif self.cell.dim == 2 and self.deg == 3:
            self._nodes = self.__lagrange_pts(self.deg)
            self._local_nodes = {0: {0: [0],
                                     1: [9],
                                     2: [3]},
                                 1: {0: [4,7],
                                     1: [8,6],
                                     2: [2,1]},
                                 2: {0: [5]}}
        elif self.cell.dim == 2 and self.deg == 4:
            self._nodes = self.__lagrange_pts(self.deg)
            self._local_nodes = {0: {0: [0],
                                     1: [14],
                                     2: [4]},
                                 1: {0: [5,9,12],
                                     1: [13,11,8],
                                     2: [3,2,1]},
                                 2: {0: [6,10,7]}}
        elif self.cell.dim == 2:
            p = self.deg
            self._nodes = self.__lagrange_pts(self.deg)
            self._local_nodes = {0: {0: [0],
                                     1: [int(binom(p+d, d)-1)],
                                     2: [p]},
                                 1: {0: [sum([p+1-i for i in range(l+1)]) for l in range(p-1)],
                                     1: list(np.flip([p+sum([p-i for i in range(l+1)]) for l in range(p-1)])),
                                     2: list(np.arange(p)[:0:-1])},
                                 2: {0: [7,12,6, 13, 9, 8, 7]}}
            foo = np.array([False]*int(binom(p+d, d)))
            for ii in self._local_nodes[0].values():
                foo[ii] = True
            for ii in self._local_nodes[1].values():
                foo[ii] = True

            self._local_nodes[2][0] = list(np.arange(int(binom(p+d,d)))[~foo])
        else:
            raise NotImplementedError()

        self._nodes_per_entity = np.zeros(self.cell.dim+1, dtype=np.int)
        for d in self.local_nodes:
            self._nodes_per_entity[d] = len(self.local_nodes[d][0])

        # Compute basis coefficients in monomial basis depending on local nodes
        V = vandermonde(self.cell, self.deg, self.nodes, self.dim)
        self._basisCoefficients = np.linalg.inv(V)


    def __lagrange_pts(self, deg):
        # generate equally spaced nodes
        #  top to bottom, left to right
        pts = []
        for i in range(deg+1):
            for j in range(deg+1-i):
                pts.append([i/deg, j/deg])

        return np.array(pts, dtype=np.double)


    def phi_eval(self, pts):
        V = vandermonde(self.cell, self.deg, pts, self.dim)
        return V@self._basisCoefficients


    def grad_phi_eval(self, pts):
        '''
        -> tensor shape = (len(pts), self.dim, cell.dim)
        '''
        dV = vandermonde_grad(self.cell, self.deg, pts, self.dim)
        gradPhi = np.empty((dV.shape[0], self.dim, self.cell.dim))
        for i in range(dV.shape[0]):  # for each point
            gradPhi[i] = self._basisCoefficients.T @ dV[i]

        return gradPhi


    @property
    def deg(self):
        return self._deg
