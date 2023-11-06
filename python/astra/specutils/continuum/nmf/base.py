
import numpy as np
import warnings
import pickle
from astra.utils import expand_path
from functools import cache
from typing import Optional, Tuple
from scipy import optimize as op
from sklearn.decomposition._nmf import _fit_coordinate_descent
from sklearn.exceptions import ConvergenceWarning

@cache
def load_components(path, P, pad=0):    
    with open(expand_path(path), "rb") as fp:
        masked_components = pickle.load(fp)
    if pad > 0:
        components = np.zeros((masked_components.shape[0], P + 2 * pad))
        components[:, pad:-pad] = masked_components
    else:
        components = masked_components
    return components

class BaseNMFSinusoidsContinuum(object):

    def __init__(
        self,
        dispersion: np.array,
        components: np.ndarray,
        deg: int,
        L: float,
        regions: Optional[Tuple[Tuple[float, float]]] = None
    ):       
        self.dispersion = dispersion
        _check_dispersion_components_shape(dispersion, components)

        self.regions = regions or [(0, dispersion.size)]
        self.components = components
        self.deg = deg
        self.L = L
        self.n_regions = len(regions)
        
        A = np.zeros(
            (self.dispersion.size, self.n_regions * self.n_parameters_per_region), 
            dtype=float
        )
        self.region_masks = region_slices(self.dispersion, self.regions)
        for i, mask in enumerate(self.region_masks):
            si = i * self.n_parameters_per_region
            ei = (i + 1) * self.n_parameters_per_region
            A[mask, si:ei] = design_matrix(dispersion[mask], self.deg, self.L).T

        self.continuum_design_matrix = A
        # TODO: Refactor to remove dependency on _dmm. Use scontinuum_deisgn_matrix instead
        #design_matrices = [
        #    design_matrix(dispersion[s], self.deg, self.L)
        #    for s in self.region_masks
        #]
        #self._dmm = (design_matrices, self.region_masks)
        return None

    @property
    def n_parameters_per_region(self):
        return 2 * self.deg + 1

    def _theta_step(self, flux, ivar, rectified_flux):        
        N, P = flux.shape
        theta = np.zeros((N, self.n_regions, self.n_parameters_per_region))
        continuum = np.nan * np.ones_like(flux)
        continuum_flux = flux / rectified_flux
        continuum_ivar = ivar * rectified_flux**2
        for i in range(N):            
            #for j, (A, mask) in enumerate(zip(*self._dmm)):
            for j, mask in enumerate(self.region_masks):
                sj, ej = (j * self.n_parameters_per_region, (j + 1) * self.n_parameters_per_region)
                A = self.continuum_design_matrix[mask, sj:ej]
                MTM = A @ (continuum_ivar[i, mask][:, None] * A.T)
                MTy = A @ (continuum_ivar[i, mask] * continuum_flux[i, mask]).T
                try:
                    theta[i, j] = np.linalg.solve(MTM, MTy)
                except np.linalg.LinAlgError:
                    if np.any(continuum_ivar[i, mask] > 0):
                        raise
                continuum[i, mask] = A.T @ theta[i, j]        

        return (theta, continuum)


    def _W_step(self, mean_rectified_flux, W, **kwargs):
        absorption = 1 - mean_rectified_flux
        use = np.zeros(mean_rectified_flux.size, dtype=bool)
        use[np.hstack(self.region_masks)] = True    
        use *= (
            np.isfinite(absorption) 
        &   (absorption >= 0) 
        &   (mean_rectified_flux > 0)
        )
        W_next, _, n_iter = _fit_coordinate_descent(
            absorption[use].reshape((1, -1)),
            W,
            self.components[:, use],
            update_H=False,
            verbose=False,
            shuffle=True
        )        
        rectified_model_flux = 1 - (W_next @ self.components)[0]
        return (W_next, rectified_model_flux, np.sum(use), n_iter)


    def get_initial_guess_by_iteration(self, flux, ivar, A=None, max_iter=32):
        C, P = self.components.shape
        ivar_sum = np.sum(ivar, axis=0)
        no_data = ivar_sum == 0
        rectified_flux = np.ones(P)
        continuum = np.ones_like(flux)
        W = np.zeros((1, C), dtype=np.float64)

        thetas, chi_sqs = ([], [])
        with warnings.catch_warnings():
            for category in (RuntimeWarning, ConvergenceWarning):
                warnings.filterwarnings("ignore", category=category)

            for iteration in range(max_iter):
                theta, continuum = self._theta_step(
                    flux,
                    ivar,
                    rectified_flux,
                )                
                mean_rectified_flux = np.sum((flux / continuum) * ivar, axis=0) / ivar_sum
                mean_rectified_flux[no_data] = 0.0
                W, rectified_flux, n_pixels_used, n_iter = self._W_step(mean_rectified_flux, W)
                chi_sqs.append(np.nansum((flux - rectified_flux * continuum)**2 * ivar))
                if iteration > 0 and (chi_sqs[-1] > chi_sqs[-2]):
                    break            

                thetas.append(np.hstack([W.flatten(), theta.flatten()]))

        return thetas[-1]


    def continuum(self, wavelength, theta):
        C, P = self.components.shape
        
        A = np.zeros(
            (wavelength.size, self.n_regions * self.n_parameters_per_region), 
            dtype=float
        )
        for i, mask in enumerate(self.region_masks):
            si = i * self.n_parameters_per_region
            ei = (i + 1) * self.n_parameters_per_region
            A[mask, si:ei] = design_matrix(wavelength[mask], self.deg, self.L).T
        
        return (A @ theta).reshape((-1, P))

    def _predict(self, theta, A_slice, C, P):
        return (1 - theta[:C] @ self.components) * (A_slice @ theta[C:]).reshape((-1, P))

            
    
    def __call__(self, theta, A_slice=None, full_output=False):
        C, P = self.components.shape
        
        if A_slice is None:
            T = len(theta)
            R = self.n_regions
            N = int((T - C) / (R * (self.n_parameters_per_region)))
            A = self.full_design_matrix(N)
            A_slice = A[:, C:]

        rectified_flux = 1 - theta[:C] @ self.components
        continuum = (A_slice @ theta[C:]).reshape((-1, P))
        flux = rectified_flux * continuum

        if not full_output:
            return flux
        return (flux, rectified_flux, continuum)
    

    def full_design_matrix(self, N):        
        C, P = self.components.shape
        R = len(self.regions)

        K = R * self.n_parameters_per_region
        A = np.zeros((N * P, C + N * K), dtype=float)
        for i in range(N):
            A[i*P:(i+1)*P, :C] = self.components.T
            A[i*P:(i+1)*P, C + i*K:C + (i+1)*K] = self.continuum_design_matrix
        return A

    def get_mask(self, ivar):
        N, P = np.atleast_2d(ivar).shape        
        use = np.zeros((N, P), dtype=bool)
        use[:, np.hstack(self.region_masks)] = True
        use *= (ivar > 0)
        return ~use

    

    def get_initial_guess_with_small_W(self, flux, ivar, A=None, small=1e-12):
        with warnings.catch_warnings():        
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            N, P = flux.shape
            if A is None:
                A = self.full_design_matrix(N)
            Y = flux.flatten()
            use = ~self.get_mask(ivar).flatten()
            result = op.lsq_linear(
                A[use],
                Y[use],
                bounds=self.get_bounds(N, [-np.inf, 0]),
            )
            
            C, P = self.components.shape
            return np.hstack([small * np.ones(C), result.x[C:]])
                    

    def get_initial_guess_by_linear_least_squares_with_bounds(self, flux, ivar, A=None):
        with warnings.catch_warnings():        
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            N, P = flux.shape
            if A is None:
                A = self.full_design_matrix(N)
            Y = flux.flatten()
            use = ~self.get_mask(ivar).flatten()            
            result = op.lsq_linear(
                A[use],
                Y[use],
                bounds=self.get_bounds(N, [-np.inf, 0]),
            )
            C, P = self.components.shape
            continuum = (A[:, C:] @ result.x[C:]).reshape(flux.shape)

            mean_rectified_flux, _ = self.get_mean_rectified_flux(flux, ivar, continuum)
            W_next, *_ = self._W_step(mean_rectified_flux, result.x[:C].astype(np.float64).reshape((1, C))) #np.zeros((1, C), dtype=np.float64))        
            return np.hstack([W_next.flatten(), result.x[C:]])


    def get_mean_rectified_flux(self, flux, ivar, continuum):
        """
        Compute a mean rectified spectrum, given an estimate of the contiuum.
        
        :param flux:
            A (N, P) shape array of flux values.
        
        :param ivar:
            A (N, P) shape array of inverse variances on flux values.
        
        :param continuum: 
            A (N, P) shape array of estimated continuum fluxes.
        """
        N, P = flux.shape
        if N == 1:
            return ((flux / continuum)[0], (ivar * continuum**2)[0])
        ivar_sum = np.sum(ivar, axis=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            mean_rectified_flux = np.sum((flux / continuum) * ivar, axis=0) / ivar_sum
        no_data = ivar_sum == 0
        mean_rectified_flux[no_data] = 0.0
        mean_rectified_ivar = np.mean(ivar * continuum**2, axis=0)        
        return (mean_rectified_flux, mean_rectified_ivar)


    def fit(self, flux: np.ndarray, ivar: np.ndarray, x0=None, full_output=False, **kwargs):

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            if kwargs.get("check_shapes", True):
                flux, ivar = _check_and_reshape_flux_ivar(self.dispersion, flux, ivar)

            N, P = flux.shape
            A = self.full_design_matrix(N)
            if x0 is None:
                x0_callables = (
                    self.get_initial_guess_with_small_W, # this one is faster and seems to be less prone to getting into runtimeerrors at optimisation time
                    self.get_initial_guess_by_linear_least_squares_with_bounds,
                )
            else:
                x0_callables = [lambda *_: x0]
        
            # build a mas using ivar and region masks
            use = ~self.get_mask(ivar).flatten()
            sigma = ivar**-0.5
            
            C, P = self.components.shape
            A_slice = A[:, C:]
            
            def f(_, *params):
                
                return self._predict(params, A_slice=A_slice, C=C, P=P).flatten()[use]
                #chi2 = (r - flux.flatten()[use]) * ivar.flatten()[use]
                #print(np.nansum(chi2))#, *params)
                #return r
                
            
            for x0_callable in x0_callables:
                x0 = x0_callable(flux, ivar, A)
                try:                            
                    p_opt, cov = op.curve_fit(
                        f,
                        None,
                        flux.flatten()[use],
                        p0=x0,
                        sigma=sigma.flatten()[use],
                        bounds=self.get_bounds(flux.shape[0])
                    )
                except RuntimeError:
                    continue
                else:
                    break
            else:
                raise RuntimeError(f"Optimization failed")
            
            model_flux, rectified_model_flux, continuum = self(p_opt, full_output=True)

            chi2 = ((model_flux - flux)**2 * ivar).flatten()
            chi2[~use] = np.nan
            rchi2 = np.sum(chi2[use]) / (use.sum() - p_opt.size - 1)

            # the first N_component values of p_opt are the W coefficients
            # the remaining values are the theta coefficients
            result = dict(
                W=p_opt[:self.components.shape[0]],
                theta=p_opt[self.components.shape[0]:],
                model_flux=model_flux,
                rectified_model_flux=rectified_model_flux,
                continuum=continuum,
                mask=~use,
                pixel_chi2=chi2,
                rchi2=rchi2
            )

            if full_output:
                return (continuum, result)
            else:
                return continuum


    def get_bounds(self, N, component_bounds=(0, +np.inf)):
        C, P = self.components.shape          
        A = N * self.n_regions * (self.n_parameters_per_region)

        return np.vstack([
            np.tile(component_bounds, C).reshape((C, 2)),
            np.tile([-np.inf, +np.inf], A).reshape((-1, 2))
        ]).T            



def region_slices(dispersion, regions):
    slices = []
    for region in regions:
        slices.append(np.arange(*dispersion.searchsorted(region), dtype=int))
    return slices


def _check_and_reshape_flux_ivar(dispersion, flux, ivar):
    P = dispersion.size
    flux, ivar = (np.atleast_2d(flux), np.atleast_2d(ivar))
    N1, P1 = flux.shape
    N2, P2 = ivar.shape

    assert (N1 == N2) and (P1 == P2), "`flux` and `ivar` do not have the same shape"
    assert (P == P1), f"Number of pixels in flux does not match dispersion array ({P} != {P1})"

    bad_pixel = (
        (~np.isfinite(flux))
    |   (~np.isfinite(ivar))
    |   (flux <= 0)
    )
    flux[bad_pixel] = 0
    ivar[bad_pixel] = 0
    return (flux, ivar)


def _check_dispersion_components_shape(dispersion, components):
    P = dispersion.size
    assert dispersion.ndim == 1, "Dispersion must be a one-dimensional array." 
    C, P2 = components.shape
    assert P == P2, "`components` should have shape (C, P) where P is the size of `dispersion`"


def design_matrix(dispersion: np.array, deg: int, L: float) -> np.array:
    scale = 2 * (np.pi / L)
    return np.vstack(
        [
            np.ones_like(dispersion).reshape((1, -1)),
            np.array(
                [
                    [np.cos(o * scale * dispersion), np.sin(o * scale * dispersion)]
                    for o in range(1, deg + 1)
                ]
            ).reshape((2 * deg, dispersion.size)),
        ]
    )





