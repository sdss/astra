
import logging as logger
from astropy.utils.misc import dtype_bytes_or_chars
import numpy as np
import pickle
from scipy import interpolate
from tqdm import tqdm
import warnings

from .photosphere import Photosphere

class NewPhotosphereInterpolator:

    default_interpolate_log_quantities = ("RHOX", "XNE", "numdens_other", "Pe", "Pg", "KappaRoss", "Density", "Depth")

    def __init__(self, photospheres, grid_keywords=None, decimals=5, method="linear", rescale=True, basis_column_name="lgTauR", interpolate_log_quantities=default_interpolate_log_quantities):
        """
        Create a new Photosphere interpolator.
        
        :param photospheres:
            A list of :class:`grok.Photosphere` objects to use for interpolation.
        
        :param grid_keywords: [optional]
            Supply the metadata keywords from each :class:`grok.Photosphere` that should be used
            for interpolation. If `None` is given, these will be taken from the `grid_keywords`
            metadata keyword in the first photosphere.
        
        :param decimals: [optional]
            The number of decimal places to round the grid points to. If `None` is given,
            no rounding will occur. 
            
            An example where this is useful is for surface gravity (logg) in MARCS models. The
            MARCS model files contain gravity, and we calculate log(g). That causes some rounding
            issues, which becomes a problem when we are exactly matching against atmospheres.
        """
        self.photospheres = list(photospheres)
        if (N := len(self.photospheres)) <= 1:
            raise ValueError(f"Need more than {N} photospheres.")

        # Build the grid of points.
        if grid_keywords is None:
            grid_keywords = tuple(self.photospheres[0].meta["grid_keywords"])
        self.basis_column_name = basis_column_name
        self.grid_keywords = grid_keywords
        self.decimals = decimals
        self.method = method
        self.rescale = rescale
        self.interpolate_log_quantities = interpolate_log_quantities or ()

        self._offsets = {}
        for column_name in self.interpolate_log_quantities:
            self._offsets[column_name] = np.min([np.min(p[column_name]) for p in self.photospheres]) - 1                
            for photosphere in self.photospheres:
                photosphere[f"__{column_name}"] = photosphere[column_name]
                
        return None

    '''
    def get_basis(self, photosphere):
        """
        Return τ_Rossland, which is used as the basis for interpolating other photospheric quantities.
        """
        # The `readmarcs.f` file provided at https://marcs.astro.uu.se/documents/auxiliary/readmarcs.f
        # indicates that:
        # -> the second column is log(tau(Rosseland)), and 
        # -> the fourth column is depth [cm], where depth = 0.0 @ tau(Rosseland) = 1.0
        # This seems to be self-consistent, because:
        assert photosphere["Depth"][np.exp(photosphere["lgTauR"]) == 1.0] == 0.0

        # In the `interpol_marcs.tar.gz` tar ball at https://marcs.astro.uu.se/documents/auxiliary/interpol_marcs.tar.gz
        # They say that `tau_Rosseland` is the basis used for interpolating quantities.
        # But in line 863 of `interpol_modeles.f` they read the second column in as `tauR` (not log(tauR))!
        # So I think the `interpol_modeles.f` code is actually using `log(tauR)` as the interpolation basis.
        return photosphere["lgTauR"]

    def set_basis(self, photosphere, values):
        photosphere["lgTauR"] = values
        return None
    '''

    #@property
    #def basis_column_name(self):
    #    return "lgTauR"
    
    @property
    def grid_points(self):
        """
        An `N x D` array of grid points.

        Here `N` is the number of photospheres and `D` is the number of dimensions in the grid.
        """
        try:
            return self._grid_points
        except AttributeError:
            self._grid_points = np.array([
                [p.meta[k] for k in self.grid_keywords] for p in self.photospheres
            ])
            
            # Apply rounding before checking for duplicates.
            if self.decimals is not None:
                self._grid_points = np.round(self._grid_points, self.decimals)

            # Check for duplicates.
            remove_indices = []
            for i, column in enumerate(self._grid_points.T):
                if np.unique(column).size == 1:
                    # Warn, then remove.
                    warnings.warn(
                        f"Column index {i} ({self.grid_keywords[i]}) only has a single value: {column[0]}. "
                        f"Excluding it from interpolator dimensions."
                    )
                    remove_indices.append(i)
            
            if remove_indices:
                self.grid_keywords = tuple([kw for i, kw in enumerate(self.grid_keywords) if i not in remove_indices])
                N, D = self._grid_points.shape
                mask = np.ones(D, dtype=bool)
                mask[remove_indices] = False
                self._grid_points = self._grid_points[:, mask]
            
            unique = np.unique(self._grid_points, axis=0)
            if self._grid_points.shape != unique.shape:

                # Get an example.
                p_ = self._grid_points.view([('', self._grid_points.dtype)] * self._grid_points.shape[1])
                u_ = unique.view([('', unique.dtype)] * unique.shape[1])

                for each in u_:
                    match = (each == p_)
                    if sum(match) > 1:
                        example = f"Indices {tuple(np.where(match)[0])} have the same grid parameters: {each[0]}."
                        break
                        
                raise ValueError(
                    "There are duplicate points specified. It's likely that the photospheres have "
                    "additional values in the library that are not accounted for in the `grid_keywords` "
                    "meta. For example: the library of photospheres includes (teff, logg, fe_h, alpha_fe) "
                    "for each photosphere, but in the meta `grid_keywords` for each photosphere it is only"
                    "returning (teff, logg, fe_h) so there are multiple photospheres at each point. "
                    "For example:\n\n" + example + "\n\n "
                    "You can override the default `grid_keywords` when initiating the `PhotosphereInterpolator.`"
                )

        return self._grid_points


    def neighbour_indices(self, x, exclusion_mask=None, allow_exact_match=None):
        """
        Return the neighbouring indices to the point `x`.

        :param x: 
            A `D`-dimensional point, where the `grid_points` has dimensionality `N x D`.

        :param exclusion_mask: [optional]
            A mask to exclude grid points from interpolation. This is useful when evaluating
            the accuracy of the interpolation, but is not necessary for typical usage.
            If given, this should be a `D` length array, where `True` indicates that the 
            item should be excluded (ignored) and `False` indicates that the item should be
            used in interpolation.

            If `exclusion_mask` is `None`, and the requested point is a point in the grid,
            then no interpolation will occur: the exact grid point will be returned. 
            If `exclusion_mask` is given, then the returned photosphere will *always*
            be interpolated, even if the requested point is a grid point, and irrespective
            of whether that grid point is masked or not.

        :param allow_exact_match: [optional]
            If `True`, then if the value in one dimension exists in the grid, we will slice
            down that dimension to reduce the dimensionality of interpolation. If `False`
            then we will never do this. If `None`, then this defaults to `True` if
            `exclusion_mask` is `None` (e.g., doing normal interpolations), and defaults to
            `False` if an `exclusion_mask` is given (e.g., probably doing LOO-CV).

            An example to help explain this. If we are interpolating to the point:
            
                [4775, 3.5, 0]
            
            And we have supplied an exclusion mask to exclude this point. We're doing this
            because we want to evaluate the accuracy of the interpolation. When we supply
            the exclusion mask, the exact point no longer exists in the grid, but along each
            dimension there is a grid point that is equal to the requested point. So if we
            allowed exact matches, we would end up with no points to interpolate. 

            Instead, if we disallow exact matches, then we force the interpolation to be in
            all `D` dimensions, but we force the cube around the point to be larger than
            usual. Conceptually, this is what we want to do because it's more "honest".
        """
        if allow_exact_match is None:
            allow_exact_match = exclusion_mask is None

        if exclusion_mask is None:
            exclusion_mask = np.zeros(self.grid_points.shape[0], dtype=bool)

        grid_points = self.grid_points[~exclusion_mask]

        limits = np.vstack([f(grid_points, axis=0) for f in (np.min, np.max)])

        N, D = grid_points.shape
        mask = np.ones(N, dtype=bool)
        exact = np.any(grid_points == x, axis=0)
        
        for j, (xj, grid_slice, is_exact) in enumerate(zip(x, grid_points.T, exact)):
            if allow_exact_match and is_exact:
                mask &= (grid_slice == xj)
                continue

            # Find the nearest above and below.
            unique_diffs = np.sort(np.unique(grid_slice - xj))
            index = unique_diffs.searchsorted(0)

            lower, upper = limits.T[j]
            if not (lower <= xj <= upper) or index == 0:
                raise ValueError(f"{self.grid_keywords[j]}={xj:.2f} is on or outside the grid boundary: ({lower:.2f}, {upper:.2f})")

            si = index - 1
            ei = index + 1

            if unique_diffs[index] == 0:
                # We're exactly on the grid point.
                assert not allow_exact_match, "How did we get here?"
                # Instead we should take the left and right neighbour, not exactly this value.
                ei += 1

            left_neighbour, *_, right_neighbour = (unique_diffs[si:ei] + xj)
            # Do within a grid, or exact values?
            # Doing it with exact values only ends up giving you far fewer CV points, and the errors in
            # depth (and other quantities) are much larger.
            #mask &= ((grid_slice == right_neighbour) | (grid_slice == left_neighbour))
            mask &= ((grid_slice <= right_neighbour) & (grid_slice >= left_neighbour))
        
        D_eff = sum(~exact) if allow_exact_match else D
        E = 2**D_eff # expected number of neighbours for linear interpolation.
        N = mask.sum()
        if N < E:
            d = ", ".join([self.grid_keywords[i] for i, is_exact in enumerate(exact) if not is_exact or not allow_exact_match])
            raise ValueError(f"Expected at least {E} nearby models to interpolate in {D_eff} dimensions ({d}), but found {N}.")

        # Convert the indices back to the original frame, without any exclusion mask applied.
        return np.where(~exclusion_mask)[0][mask]


    def _loocv_columns(self, column_names, index, alphas=None):

        xi = self.grid_points[index]
        exclusion_mask = self.exclusion_mask(xi)
    
        common_basis, _, interpolated_quantities, __ = self.interpolate_columns(
            xi, 
            column_names=column_names,
            exclusion_mask=exclusion_mask,
            alphas=alphas
        )
        expected = np.array([self.photospheres[index][cn].data for cn in column_names])
        actual = interpolated_quantities
        return (expected, actual, percent(expected, actual))


    def check_point(self, point):
        # We do self.grid_points here to ensure that it is computed, and to check that if there
        # are dimensions that we don't need to interpolate over, then not to worry about them.
        # So although it looks like this line is not needed, it is for the first time interpolation
        # is called.
        self.grid_points
    
        missing_keys = set(self.grid_keywords).difference(point)
        if missing_keys:
            raise ValueError(f"Missing keyword arguments: {', '.join(missing_keys)}")
    
        xi = np.array([point[k] for k in self.grid_keywords])

        lower, upper = (np.min(self.grid_points, axis=0), np.max(self.grid_points, axis=0))
        is_lower, is_upper = (xi < lower, xi > upper)
        if np.any(is_lower) or np.any(is_upper):
            is_bad = is_lower + is_upper
            indices = np.where(is_bad)[0]
            # Make a nice error message.
            message = "Point is outside the boundaries:\n"
            for index in indices:
                message += f"- {self.grid_keywords[index]} = {xi[index]:.2f} is outside the range ({lower[index]:.2f}, {upper[index]:.2f})\n"
            message = message.rstrip()
            raise ValueError(message)
        
        return xi


    def __call__(self, exclusion_mask=None, **point):
        """
        Interpolate a photospheric structure at the given stellar parameters.

        :param method: [optional]
            The interpolation method to supply to `scipy.interpolate.griddata`.
            See `scipy.interpolate.griddata` for more information.
        
        :param rescale: [optional]
            Rescale the quantities before interpolation.
            See `scipy.interpolate.griddata` for more information.
        
        :param exclusion_mask: [optional]
            A mask to exclude grid points from interpolation. This is useful when evaluating
            the accuracy of the interpolation, but is not necessary for typical usage.
            If given, this should be a `D` length array, where `True` indicates that the 
            item should be excluded (ignored) and `False` indicates that the item should be
            used in interpolation.

            If `exclusion_mask` is `None`, and the requested point is a point in the grid,
            then no interpolation will occur: the exact grid point will be returned. 
            If `exclusion_mask` is given, then the returned photosphere will *always*
            be interpolated, even if the requested point is a grid point, and irrespective
            of whether that grid point is masked or not.

        :param **point:
            The point to interpolate at, where the keys correspond to the 
            `PhotosphereInterpolator.grid_keywords`.
        """

        print(f"in __call__")
        xi = self.check_point(point)

        # Check for an exact match.
        if exclusion_mask is None and np.any(grid_index := np.all(self.grid_points == xi, axis=1)):
            return self.photospheres[np.where(grid_index)[0][0]]

        common_basis, column_names, interpolated_quantities, meta = self.interpolate_columns(xi, exclusion_mask=exclusion_mask)

        # create a photosphere from these data.
        photosphere = Photosphere(
            data=interpolated_quantities.T,
            names=column_names,
            meta=meta
        )
        photosphere[self.basis_column_name] = common_basis

        # Fix the `k` column, if it exists.
        try:
            photosphere["k"] = np.round(photosphere["k"]).astype(int)
        except:
            None

        # Require physical consistency between columns.
        # TODO: Once we have decided what leads to the lowest overall error.

        # Update the meta keywords for interpolated properties.
        for key in self.grid_keywords:
            photosphere.meta[key] = point[key]

        for key in photosphere.dtype.names:
            if key.startswith("__"):
                photosphere[key[2:]] = photosphere[key].data.copy()
                del photosphere[key]

        return photosphere

    def _point_to_array(self, point):
        N, D = self.grid_points.shape
        if isinstance(point, dict):
            return np.array([point[k] for k in self.grid_keywords])
        else:
            if len(point) != D:
                raise ValueError(f"Expected {D} points for point {point}, not {len(point)} ({len(point)} != {D})")
            return np.array(point)


    def exclusion_mask(self, xi):

        exclusion_mask = np.zeros(self.grid_points.shape[0], dtype=bool)
        xi = self._point_to_array(xi)
        exclusion_mask[np.all(xi == self.grid_points, axis=1)] = True
        return exclusion_mask
        

    def estimate_interpolation_error(self, point, column_names=None, alphas=None):
        """
        Estimate the error in interpolation by leave-one-out cross-validation.
        
        Given a point somewhere within the bounds of the grid, this will find the nearest grid point 
        in euclidian (L2) distance and estimate the error in interpolated quantities by leaving out
        that grid point, and interpolating to it.

        :param point:
            A dictionary or array of the point where to estimate the interpolation error.
        
        :param column_names: [optional]
            The column names to estimate the interpolation error for. If `None` is given then this
            will default to all columns.

        :param alphas: [optional]
            The exponents to use for interpolation.
        
        :returns:
            A dictionary where column names are keys, and values are the estimated percent error in
            interpolation, at each depth in the photosphere.
        """

        xi = self._point_to_array(point)
        # Get nearest point in grid. Use that for LOOCV.
        grid_lower = np.min(self.grid_points, axis=0)
        grid_upper = np.max(self.grid_points, axis=0)

        grid_norm = (self.grid_points - grid_lower) / (grid_upper - grid_lower)
        xi_norm = (xi - grid_lower) / (grid_upper - grid_lower)

        distance = np.sum((grid_norm - xi_norm)**2, axis=1)
        closest_index = np.argmin(distance)
        
        column_names = column_names or list(self.photospheres[0].dtype.names) 
        expected, actual, percent = self._loocv_columns(column_names, closest_index, alphas=alphas)

        return (dict(zip(column_names, percent)), xi, column_names, expected, actual)


    def interpolate_columns(self, xi, column_names=None, exclusion_mask=None, alphas=None):

        column_names = column_names or list(self.photospheres[0].dtype.names)

        neighbour_indices = self.neighbour_indices(xi, exclusion_mask=exclusion_mask)

        # Protect Qhull from columns with a single value.
        cols = _protect_qhull(self.grid_points[neighbour_indices])  
            
        # Create a common basis for the interpolation.
        xi = xi[cols].reshape(1, len(cols))
        grid_lower = np.min(self.grid_points, axis=0)[cols]
        grid_upper = np.max(self.grid_points, axis=0)[cols]

        # TODO: Re-scale to the (min, max) of the grid, or to be (0, 1), or to be (-1, +1)?
        xi = (xi - grid_lower) / (grid_upper - grid_lower)
        points = (self.grid_points[neighbour_indices][:, cols] - grid_lower) / (grid_upper - grid_lower)

        if alphas is not None:
            alphas = np.array(alphas)
            xi = np.power(xi, alphas[cols])
            points = np.power(points, alphas[cols])

        values = np.array([self.photospheres[idx][self.basis_column_name] for idx in neighbour_indices])

        kwds = {
            "xi": xi,
            "points": points,
            "values": values,
            "method": self.method,
            "rescale": False
        }

        if np.all(np.all(values == values[0], axis=1)):
            common_basis = values[0]
        else:
            common_basis = interpolate.griddata(**kwds)
            assert np.isfinite(common_basis).all()
        
        # At the neighbouring N points, create splines of all the values
        # with respect to their own opacity scales, then calcualte the 
        # photospheric quantities on the common opacity scale.
        C = len(column_names)
        N, D = kwds["values"].shape

        resampled_neighbour_quantities = np.empty((N, C, D))
        for i, column_name in enumerate(column_names):
            for j, ni in enumerate(neighbour_indices):
                v = self.photospheres[ni][column_name]
                if column_name.startswith("__"):
                    v = np.log10(v - self._offsets[column_name[2:]])
                    assert np.all(np.isfinite(v))
                tk = interpolate.splrep(
                    self.photospheres[ni][self.basis_column_name],
                    v,
                )
                resampled_neighbour_quantities[j, i] = interpolate.splev(common_basis.flatten(), tk)
    
        interpolated_quantities = np.zeros((C, D))
        for i, column_name in enumerate(column_names):
            if np.all(resampled_neighbour_quantities[:, i][0] == resampled_neighbour_quantities[:, i]):
                z = resampled_neighbour_quantities[0, i]
            else:
                z = interpolate.griddata(
                    xi=xi,
                    points=points,
                    values=resampled_neighbour_quantities[:, i],
                    method=self.method,
                    rescale=False
                )

            if column_name.startswith("__"):
                cn = column_name[2:]
                z = 10**z + self._offsets[cn]
            interpolated_quantities[i] = z

        # Get meta from neighbours.
        meta = self.photospheres[neighbour_indices[0]].meta.copy()

        return (common_basis, column_names, interpolated_quantities, meta)


    def plot(self, point, label=None):
    
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        photosphere = self(**point)

        errors, errors_xi, errors_column_names, errors_expected, errors_actual = self.estimate_interpolation_error(point)

        ni = self.neighbour_indices(self._point_to_array(point))
        columns = [
            (self.basis_column_name, "RHOX", False, False),
            (self.basis_column_name, "lgTau5", False, False),
            (self.basis_column_name, "__Pe", False, True),
            (self.basis_column_name, "T", False, True),
            (self.basis_column_name, "__XNE", False, True),
            (self.basis_column_name, "__numdens_other", False, True),
            (self.basis_column_name, "__Density", False, True),
            (self.basis_column_name, "__Depth", False, False)
        ]
        K = int(len(columns) / 2)
        ncols, nrows = (K + 2, 2)
        width_ratio = 2
        width_ratios = np.hstack([width_ratio, *np.ones(K), width_ratio])
        fig, axes = plt.subplots(figsize=(15.25, 3.25), ncols=ncols, nrows=nrows, gridspec_kw=dict(width_ratios=width_ratios))
        for ax in np.array(fig.axes):
            if ax.get_subplotspec().is_last_col() or ax.get_subplotspec().is_first_col():
                ax.remove()
        axes = list(fig.axes) 
        param_ax = fig.add_subplot(GridSpec(ncols=ncols, nrows=1, width_ratios=width_ratios)[0])
        error_ax = fig.add_subplot(GridSpec(ncols=ncols, nrows=1, width_ratios=width_ratios)[-1])

            

        ylabels = [ylabel for _, ylabel, *__ in columns]
        colors = dict(zip(ylabels, plt.rcParams['axes.prop_cycle'].by_key()['color']))

        # use line styling for logg.
        unique_logg = np.unique(self.grid_points[ni].T[1])
        line_styles = dict(zip(unique_logg, ["-", ":", "--", "-.",]))

        for ax, (xlabel, ylabel, semilogx, semilogy) in zip(axes, columns):

            if semilogy and min(photosphere[ylabel.lstrip("_")]) < 1:
                offset = -np.min(np.hstack([
                    np.min(photosphere[ylabel.lstrip("_")]),
                    *[np.min(self.photospheres[index][ylabel]) for index in ni]
                ]))
            else:
                offset = 0

            x = photosphere[xlabel.lstrip("_")]
            y = offset + photosphere[ylabel.lstrip("_")]
            y_error = y * np.abs(errors[ylabel])/100

            ax.plot(x, y, c=colors[ylabel])
            ax.fill_between(
                x, 
                y - y_error,
                y + y_error,
                facecolor=colors[ylabel],
                alpha=0.3
            )

            ax.set_xlabel(xlabel)
            if offset > 0:
                ax.set_ylabel(f"{ylabel} + offset")
            else:
                ax.set_ylabel(ylabel)

            for index in ni:
                ax.plot(
                    self.photospheres[index][xlabel],
                    offset + self.photospheres[index][ylabel],
                    c="k",
                    #ls=line_styles[self.grid_points[index][1]],
                    alpha=0.25
                )
                if semilogy and offset > 0:
                    assert np.all((offset + self.photospheres[index][ylabel]) >= 0)
            
            if semilogx and semilogy:
                ax.loglog()
            elif semilogx:
                ax.semilogx()
            elif semilogy:
                ax.semilogy()

        plot_columns = set(np.array(columns)[:, [0, 1]].flatten()).difference([self.basis_column_name])
        for column in plot_columns:
            error_ax.plot(
                photosphere[self.basis_column_name],
                errors[column],
                label=column.lstrip("_"),
                c=colors[column]
            )
        error_ax.axhline(0, c="#666666", ls=":", zorder=-1, lw=0.5)
        error_ax.set_xlabel(self.basis_column_name)
        error_ax.set_ylabel(f"Estimated error [%]")

        if self.grid_points.shape[1] > 2:    
            unique_z = np.sort(np.unique(self.grid_points[ni].T[2]))
            radius_x, radius_y = np.ptp(self.grid_points[ni], axis=0)[:2] / 15
            thetas = np.linspace(0, 2 * np.pi, 5)

            offset_x = radius_x * np.array([np.cos(thetas[np.where(unique_z == z)[0][0]]) for z in self.grid_points[ni].T[2]])
            offset_y = radius_y * np.array([np.sin(thetas[np.where(unique_z == z)[0][0]]) for z in self.grid_points[ni].T[2]])

            param_ax.scatter(
                self.grid_points[ni].T[0],
                self.grid_points[ni].T[1],
                facecolor="k",
                s=1,
            )

        else:
            unique_z = []
            offset_x, offset_y = (0, 0)

        param_ax.scatter(
            self.grid_points[ni].T[0] + offset_x,
            self.grid_points[ni].T[1] + offset_y,
            facecolor="k",
            alpha=0.5,
            s=5,
        )

        label = label or f",  ".join([f"{k}: {v:.2f}" for k, v in point.items()])
        xi, yi = (point[self.grid_keywords[0]], point[self.grid_keywords[1]])
        param_ax.scatter([xi], [yi], facecolor="k")
        param_ax.annotate(
            label,
            xy=(xi, yi),
            xycoords="data",
            xytext=(xi + np.ptp(self.grid_points[ni].T[0])/5, yi),
            textcoords="data",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )

        xi = np.min(self.grid_points[ni].T[0])
        yi = np.max(self.grid_points[ni].T[1])

        for j, z in enumerate(unique_z):
            param_ax.text(
                xi + radius_x * np.cos(thetas[j]) + 0.5 * radius_x,
                yi + radius_y * np.sin(thetas[j]),# + radius_y,
                f"{z:.2f}",
                verticalalignment="center",
                horizontalalignment="left",
            )

        param_ax.set_xlabel(self.grid_keywords[0])
        param_ax.set_ylabel(self.grid_keywords[1])

        fig.tight_layout()
        return fig



def percent(expected, actual):
    p = (actual - expected) / expected
    p[expected == 0] = 0
    return 100 * p

import itertools


def calculate_alpha(interpolator, column_name, alpha_iter):

    N, D = interpolator.grid_points.shape

    all_alphas = []
    for alphas in itertools.combinations_with_replacement(alpha_iter, D):
        all_alphas.append(alphas)
    all_alphas = np.array(all_alphas)

    A, D = all_alphas.shape
    L = len(interpolator.photospheres[0])
    shape = (A, N, L)
    all_percent = np.nan * np.ones(shape)
    all_expected = np.nan * np.ones(shape)
    all_actual = np.nan * np.ones(shape)


    for n in tqdm(range(N)):
        for a, alphas in enumerate(all_alphas):
            try:
                expected, actual, percent = interpolator._loocv_columns((column_name, ), n, alphas=alphas)
            except ValueError:
                continue
            except:
                raise
            all_actual[a, n] = actual
            all_expected[a, n] = expected
            all_percent[a, n] = percent
            
    return (all_expected, all_actual, all_percent, all_alphas)
    

    

def loo_cv(interpolator, raise_exceptions=False):
    """
    Perform leave-one-out cross-validation on the grid.
    """
    from tqdm import tqdm

    column_names = list(interpolator.photospheres[0].dtype.names)

    # Create a big-ass array of all the quantities we are going to compare.
    N, D = interpolator.grid_points.shape
    C = len(column_names)
    Q = len(interpolator.photospheres[0])

    actual = np.nan * np.ones((N, C, Q))
    expected = np.nan * np.ones((N, C, Q))

    failed_indices = []
    
    for n, grid_point in tqdm(enumerate(interpolator.grid_points), total=N):

        for c, column_name in enumerate(column_names):
            expected[n, c] = interpolator.photospheres[n][column_name]
        
        if isinstance(interpolator, NewPhotosphereInterpolator):
            try:
                _, actual[n], __ = interpolator._loocv_columns(column_names, n)
            except AssertionError:
                raise 
            except:
                #logger.exception(f"An exception occured interpolating index {n} {grid_point}")
                failed_indices.append(n)
                actual[n][:] = np.nan
                if raise_exceptions:
                    raise

        else:
            exclusion_mask = np.zeros(N, dtype=bool)
            exclusion_mask[n] = True

            point = dict(zip(interpolator.grid_keywords, grid_point))
            try:
                photosphere = interpolator(exclusion_mask=exclusion_mask, **point)
            except AssertionError:
                raise
            except:
                failed_indices.append(n)
                actual[n][:] = np.nan
                if raise_exceptions:
                    raise
            else:
                for c, column_name in enumerate(column_names):
                    actual[n, c] = photosphere[column_name].data

    outcome = np.ones(N, dtype=bool)
    outcome[failed_indices] = False
    return (column_names, expected, actual, percent(expected, actual), outcome)



class PhotosphereInterpolator(object):

    def __init__(self, photospheres, grid_keywords=None,  neighbours=30, method="linear", rescale=True, interpolate_log_quantities=None):

        self.photospheres = list(photospheres)

        if len(self.photospheres) <= 1:
            raise ValueError(f"Need more photospheres than that ({len(self.photospheres)} given).")

        # Build the grid of points.
        if grid_keywords is None:
            grid_keywords = tuple(self.photospheres[0].meta["grid_keywords"])
        self.neighbours = neighbours
        self.grid_keywords = grid_keywords
        self.method = method
        self.rescale = rescale
        self.interpolate_log_quantities = interpolate_log_quantities or ()
        return None
        

    @property
    def opacity_column_name(self):
        keys = ("RHOX", "tau")
        for key in keys:
            if key in self.photospheres[0].dtype.names:
                return key
        raise KeyError(f"Cannot identify opacity column name. Tried: {', '.join(keys)}")



    @property
    def grid_points(self):
        try:
            return self._grid_points
        except AttributeError:
            self._grid_points = np.array([
                [p.meta[k] for k in self.grid_keywords] for p in self.photospheres
            ])

            # Check for duplicates.
            remove_indices = []
            for i, column in enumerate(self._grid_points.T):
                if np.unique(column).size == 1:
                    # Warn, then remove.
                    warnings.warn(
                        f"Column index {i} ({self.grid_keywords[i]}) only has a single value: {column[0]}. "
                        f"Excluding it from interpolator dimensions."
                    )
                    remove_indices.append(i)
            
            if remove_indices:
                self.grid_keywords = tuple([kw for i, kw in enumerate(self.grid_keywords) if i not in remove_indices])
                N, D = self._grid_points.shape
                mask = np.ones(D, dtype=bool)
                mask[remove_indices] = False
                self._grid_points = self._grid_points[:, mask]
            
            unique = np.unique(self._grid_points, axis=0)
            if self._grid_points.shape != unique.shape:

                # Get an example.
                p_ = self._grid_points.view([('', self._grid_points.dtype)] * self._grid_points.shape[1])
                u_ = unique.view([('', unique.dtype)] * unique.shape[1])

                for each in u_:
                    match = (each == p_)
                    if sum(match) > 1:
                        example = f"Indices {tuple(np.where(match)[0])} have the same grid parameters: {each[0]}."
                        break
                        
                raise ValueError(
                    "There are duplicate points specified. It's likely that the photospheres have "
                    "additional values in the library that are not accounted for in the `grid_keywords` "
                    "meta. For example: the library of photospheres includes (teff, logg, fe_h, alpha_fe) "
                    "for each photosphere, but in the meta `grid_keywords` for each photosphere it is only"
                    "returning (teff, logg, fe_h) so there are multiple photospheres at each point. "
                    "For example:\n\n" + example + "\n\n "
                    "You can override the default `grid_keywords` when initiating the `PhotosphereInterpolator.`"
                )

        return self._grid_points


    def write(self, path):
        """
        Write this photosphere interpolator to disk. This will store all models and their metadata.

        :param path:
            The path to store the interpolator.
        """

        # Photosphere information first.
        column_names = self.photospheres[0].dtype.names

        N = len(self.photospheres)
        C = len(column_names)
        D = len(self.photospheres[0][column_names[0]])

        structure = np.nan * np.ones((N, C, D))
        for i in range(N):
            for j, column_name in enumerate(column_names):
                # In rare circumstances, a model can have one fewer depth points than it's neighbours.
                _structure = self.photospheres[i][column_name]
                structure[i, j, :len(_structure)] = _structure
                
        contents = {
            "points": self.grid_points,
            "column_names": column_names,
            "structure": structure,
            "meta": [p.meta for p in self.photospheres],
        }
        raise NotImplementedError()

        with open(path, "wb") as fp:
            pickle.dump(contents, fp)
        
        return None


    @classmethod
    def read(cls, path):
        """
        Read a photosphere interpolator from disk.
        
        :param path:
            The path where the library of models is stored.
        """

        with open(path, "rb") as fp:
            contents = pickle.load(fp)

        interpolator = cls([], None)
        interpolator._points = contents["points"]
        
        raise a





    def __call__(self, full_output=False, exclusion_mask=None, **point):
        """
        Interpolate a photospheric structure at the given stellar parameters.
        """

        grid_points = self.grid_points
        if exclusion_mask is not None:
            grid_points = grid_points[~exclusion_mask]

        missing_keys = set(self.grid_keywords).difference(point)
        if missing_keys:
            raise ValueError(f"Missing keyword arguments: {', '.join(missing_keys)}")
        
        interpolate_column_names = [n for n in self.photospheres[0].dtype.names if n != self.opacity_column_name]

        xi = np.array([point[k] for k in self.grid_keywords])

        lower, upper = (np.min(grid_points, axis=0), np.max(grid_points, axis=0))
        is_lower, is_upper = (xi < lower, xi > upper)
        if np.any(is_lower) or np.any(is_upper):
            is_bad = is_lower + is_upper
            indices = np.where(is_bad)[0]
            bad_values = xi[indices]
            raise ValueError(
                f"Point is outside the boundaries: {bad_values} (indices {indices}) outside bounds "
                f"(lower: {lower[indices]}, upper: {upper[indices]})"
            )

        grid_index = np.all(grid_points == xi, axis=1)
        if np.any(grid_index):
            grid_index = np.where(grid_index)[0][0]
            return self.photospheres[grid_index]

        distances = np.sum(((xi - grid_points) / np.ptp(grid_points, axis=0))**2, axis=1)
        neighbour_indices = distances.argsort()[:self.neighbours]

        # Protect Qhull from columns with a single value.
        cols = _protect_qhull(grid_points[neighbour_indices])  
        
        kwds = {
            "xi": xi[cols].reshape(1, len(cols)),
            "points": grid_points[neighbour_indices][:, cols],
            "values": np.array([self.photospheres[ni][self.opacity_column_name] for ni in neighbour_indices]),
            "method": self.method,
            "rescale": self.rescale
        }
        common_opacity_scale = interpolate.griddata(**kwds)

        # At the neighbouring N points, create splines of all the values
        # with respect to their own opacity scales, then calcualte the 
        # photospheric quantities on the common opacity scale.

        C = len(interpolate_column_names)
        N, D = kwds["values"].shape

        interpolated_quantities = np.zeros((C, D))
        neighbour_quantities = np.array([[self.photospheres[ni][column_name] for column_name in interpolate_column_names] for ni in neighbour_indices])
        for i, column_name in enumerate(interpolate_column_names):
            if column_name in self.interpolate_log_quantities:
                kwds["values"] = np.log10(neighbour_quantities[:, i])
            else:
                kwds["values"] = neighbour_quantities[:, i]

            z = interpolate.griddata(**kwds)
            if column_name in self.interpolate_log_quantities:
                z = 10**z
            interpolated_quantities[i] = z

        # Get meta from neighbours.
        meta = self.photospheres[neighbour_indices[0]].meta.copy()

        # create a photosphere from these data.
        photosphere = Photosphere(
            data=np.vstack([common_opacity_scale, interpolated_quantities]).T,
            names=tuple([self.opacity_column_name] + interpolate_column_names),
            meta=meta
        )

        # Update the meta keywords for interpolated properties.
        for key in self.grid_keywords:
            photosphere.meta[key] = point[key]

        if full_output:
            return (photosphere, common_opacity_scale, neighbour_quantities, neighbour_indices, interpolate_column_names)
        else:
            return photosphere


def _protect_qhull(a):
    return np.where([np.unique(a[:, i]).size > 1 for i in range(a.shape[1])])[0]
    
