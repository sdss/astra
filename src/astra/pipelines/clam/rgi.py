

from scipy.interpolate._rgi import _check_points
from scipy.interpolate._ndbspline import make_ndbspl
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import _bspl

import numpy as np

from math import prod

class CustomGridInterpolator:

    def __init__(self, points, values, k=1):
        self.grid, self._descending_dimensions = _check_points(points)
        assert not self._descending_dimensions
        self.values = values
        
        self._spline = make_ndbspl(self.grid, self.values, k)

        self.ndim = ndim = len(self._spline.t)
        # prepare k & t
        self._k = np.asarray(self._spline.k, dtype=np.dtype("long"))

        # pack the knots into a single array
        len_t = [len(ti) for ti in self._spline.t]
        self._t = np.empty((ndim, max(len_t)), dtype=float)
        self._t.fill(np.nan)
        for d in range(ndim):
            self._t[d, :len(self._spline.t[d])] = self._spline.t[d]
        self.len_t = np.asarray(len_t, dtype=np.dtype("long"))

        # tabulate the flat indices for iterating over the (k+1)**ndim subarray
        shape = tuple(kd + 1 for kd in self._spline.k)
        indices = np.unravel_index(np.arange(prod(shape)), shape)
        self._indices_k1d = np.asarray(indices, dtype=np.intp).T

        # prepare the coefficients: flatten the trailing dimensions
        self.c1 = self._spline.c.reshape(self._spline.c.shape[:ndim] + (-1,))
        self.c1r = self.c1.ravel()

        # replacement for np.ravel_multi_index for indexing of `c1`:
        self._strides_c1 = np.asarray([s // self.c1.dtype.itemsize
                                  for s in self.c1.strides], dtype=np.intp)

        self.num_c_tr = self.c1.shape[-1]  # # of trailing coefficients
        self.nu_0 = np.zeros((self.ndim, ), dtype=np.intc)


    def jacobian(self, xi):
        xi = np.ascontiguousarray(np.atleast_2d(xi))
        jac = np.empty((self.ndim, self.num_c_tr), dtype=self.c1.dtype)
        for i in range(self.ndim):
            out = np.empty(xi.shape[:-1] + (self.num_c_tr,), dtype=self.c1.dtype)
            nu = np.zeros((self.ndim, ), dtype=np.intc)
            nu[i] = 1
            _bspl.evaluate_ndbspline(xi,
                                    self._t,
                                    self.len_t,
                                    self._k,
                                    nu,
                                    True, # extrapolate
                                    self.c1r,
                                    self.num_c_tr,
                                    self._strides_c1,
                                    self._indices_k1d,
                                    out,)     
            jac[i] = out
        return jac         


    def __call__(self, xi):
        xi = np.ascontiguousarray(np.atleast_2d(xi))
        out = np.empty(xi.shape[:-1] + (self.num_c_tr,), dtype=self.c1.dtype)
        _bspl.evaluate_ndbspline(xi,
                                 self._t,
                                 self.len_t,
                                 self._k,
                                 self.nu_0,
                                 False, # extrapolate
                                 self.c1r,
                                 self.num_c_tr,
                                 self._strides_c1,
                                 self._indices_k1d,
                                 out,)                                
        return out



if __name__ == "__main__":
        
    import pickle
    from astra.utils import expand_path

    with open(expand_path("~/sas_home/continuum/20240806.pkl"), "rb") as fp:
        label_names, grid_points, W, H_clipped = pickle.load(fp)

    W = W.reshape((list(map(len, grid_points)) + [-1]))

    is_warm = (grid_points[-1] >= 4000)
    grid = grid_points[:-1] + [grid_points[-1][is_warm]]
    points = W[..., is_warm, :]

    old = RegularGridInterpolator(grid, points,
        method="slinear",
        bounds_error=False,
        fill_value=0.0 
    )

    new = CustomGridInterpolator(grid, points)

    from time import time

    N = 100000
    np.random.seed(0)
    xis = np.random.uniform(
        tuple(map(min, grid_points)),
        tuple(map(max, grid_points)),
        size=(N, 7)
    )

    t_init = time()
    for xi in xis:
        old(xi)
    t_old = time() - t_init



    t_init = time()
    for xi in xis:
        new(xi)
    t_new = time() - t_init

    for xi in xis:
        if np.all(np.isfinite(old(xi))) and np.any(old(xi) > 0):
            assert np.all(old(xi) == new(xi))


