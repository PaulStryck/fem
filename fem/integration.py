import numpy as np


class ReferenceSimplexQuadrature():
    def __init__(self, dimension, degree, pts, weights):
        self._dimension = dimension
        self._degree    = degree
        self._pts       = np.array(pts, dtype=np.double)
        self._weights   = np.array(weights, dtype=np.double)

        if self._pts.shape[0] != len(self.weights):
            raise ValueError(
                "Number of quadrature points and weights must match")

        if self._pts.shape[1] != self._dimension:
            raise ValueError("Dimension must match shape of pts")

    @property
    def weights(self):
        return self._weights

    @property
    def points(self):
        return self._pts

    def integrate(self, fn):
        '''
        :param fn: the function to integrate on this simplex. Must take an
                   np.array of shape(self.dimension,) and output a scalar value
        '''
        return np.array([fn(x) for x in self.points]) @ self.weights


def gauss_legendre_quadrature(dimension, degree):
    if dimension == 1:
        # quadrature rule for [0,1] interval

        # Gauss-legendre quadrature is of degree = 2 * npoints - 1
        npoints = int((degree + 2) / 2)

        points, weights = np.polynomial.legendre.leggauss(npoints)

        # map numpys [-1,1] interval back to [0,1]
        points = np.expand_dims((points + 1.) / 2., 1)
        weights = weights / 2.

    elif dimension == 2:
        # quadrature rule for 2d simplex (aka triangle)
        # obtained by transforming a unit square

        p1 = gauss_legendre_quadrature(1, degree + 1)
        q1 = gauss_legendre_quadrature(1, degree)

        points = np.array([(p[0], q[0] * (1 - p[0]))
                           for p in p1.points
                           for q in q1.points])

        weights = np.array([p * q * (1 - x[0])
                            for p, x in zip(p1.weights, p1.points)
                            for q in q1.weights])
    else:
        raise ValueError("Dimension not yet supported")

    return ReferenceSimplexQuadrature(dimension, degree, points, weights)

