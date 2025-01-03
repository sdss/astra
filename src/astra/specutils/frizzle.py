import finufft
import numpy as np
import numpy.typing as npt
import warnings
from pylops import LinearOperator, MatrixMult, Diagonal, Identity
from scipy.linalg import lstsq
from scipy.sparse.linalg import lsqr
from sklearn.neighbors import KDTree
from typing import Optional, Union, Tuple

def frizzle(
    λ_out: npt.ArrayLike,
    λ: npt.ArrayLike,
    flux: npt.ArrayLike,
    ivar: Optional[npt.ArrayLike] = None,
    mask: Optional[npt.ArrayLike] = None,
    flags: Optional[npt.ArrayLike] = None,
    n_modes: Optional[int] = None,
    n_uncertainty_samples: Optional[Union[int, float]] = None,
    censor_missing_regions: Optional[bool] = True,
    lsqr_kwds: Optional[dict] = None,
    finufft_kwds: Optional[float] = None,
):
    """
    Combine spectra by forward modeling.

    :param λ_out:
        The wavelengths to sample the combined spectrum on.
    
    :param λ:
        The wavelengths of the individual spectra. This should be shape (N, ) where N is the number of pixels.
    
    :param flux:
        The flux values of the individual spectra. This should be shape (N, ).
    
    :param ivar: [optional]
        The inverse variance of the individual spectra. This should be shape (N, ).
    
    :param mask: [optional]
        The mask of the individual spectra. If given, this should be a boolean array (pixels with `True` get ignored) of shape (N, ).
        The mask is used to ignore pixel flux when combining spectra, but the mask is not used when computing combined pixel flags.
    
    :param flags: [optional]
        An optional integer array of bitwise flags. If given, this should be shape (N, ).
    
    :param n_modes: [optional]
        The number of Fourier modes to use. If `None` is given then this will default to `len(λ_out)`.
    
    :param n_uncertainty_samples: [optional]
        The number of samples to use when estimating the uncertainty of the combined spectrum by Hutchinson's method. 
        If `None` is given then this will default to `0.10 * n_modes`. If a float is given then this will be interpreted as a fraction 
        of `n_modes`. If a number is given that exceeds `n_modes`, the uncertainty will be computed exactly (slow).

    :param censor_missing_regions: [optional]
        If `True`, then regions where there is no data will be set to NaN in the combined spectrum. If `False` the values evaluated
        from the model will be reported (and have correspondingly large uncertainties) but this will produce unphysical features.
        
    :param finufft_kwds: [optional]
        Keyword arguments to pass to the `finufft.Plan()` constructor.
    
    :param lsqr_kwds: [optional]
        Keyword arguments to pass to the `scipy.sparse.linalg.lsqr()` function.

        The most relevant `lsqr()` keyword to the user is `calc_var`, which will compute the variance of the combined spectrum.
        By default this is set to `True`, but it can be set to `False` to speed up the computation if the variance is not needed.
    
    :returns:
        A four-length tuple of ``(flux, ivar, flags, meta)`` where:
            - ``flux`` is the combined fluxes,
            - ``ivar`` is the variance of the combined fluxes,
            - ``flags`` are the combined flags, and 
            - ``meta`` is a dictionary.
    """

    n_modes = n_modes or len(λ_out)
    lsqr_kwds = ensure_dict(lsqr_kwds, calc_var=True)
    finufft_kwds = ensure_dict(finufft_kwds)

    λ_out, λ, flux, ivar, mask = check_inputs(λ_out, λ, flux, ivar, mask)

    λ_all = np.hstack([λ[~mask], λ_out])
    λ_min, λ_max = (np.min(λ_all), np.max(λ_all))

    # This is setting the scale to be such that the Fourier modes are in the range [0, 2π).
    small = 1e-5
    scale = (1 - small) * 2 * np.pi / (λ_max - λ_min)
    x = (λ[~mask] - λ_min) * scale
    x_star = (λ_out - λ_min) * scale
    
    C_inv_sqrt = np.sqrt(ivar[~mask])

    A = FrizzleOperator(x, n_modes, **finufft_kwds)
    C_inv = Diagonal(np.ascontiguousarray(ivar[~mask]))

    A_w = Diagonal(C_inv_sqrt) @ A
    Y_w = C_inv_sqrt * flux[~mask]
    θ, *extras = lsqr(A_w, Y_w, **lsqr_kwds)
    
    meta = dict(zip(["istop", "itn", "r1norm", "r2norm", "anorm", "acond", "arnorm", "xnorm", "var"], extras))
    
    A_star = FrizzleOperator(x_star, n_modes, **finufft_kwds)
    y_star = A_star @ θ

    # Until I know what lsqr `var` is doing..
    #ATCinvA_inv = lsqr(A.T @ C_inv @ A, np.ones(n_modes), **lsqr_kwds)[0]
    ATCinvA_inv = meta["var"]
    Op = (A_star @ Diagonal(ATCinvA_inv) @ A_star.T)

    n_uncertainty_samples = n_uncertainty_samples or 0.1
    if n_uncertainty_samples < 1:
        n_uncertainty_samples *= n_modes
        
    n_uncertainty_samples = int(n_uncertainty_samples)        
    if n_uncertainty_samples < n_modes:
        # Estimate the diagonals with Hutchinson's method.
        v = np.random.randn(n_modes, n_uncertainty_samples)
        C_inv_star = n_uncertainty_samples/np.sum((Op @ v) * v, axis=1)
    else:            
        C_inv_star = 1/np.diag(Op.todense())
    
    if censor_missing_regions:
        # Set NaNs for regions where there were NO data.
        # Here we check to see if the closest input value was more than the output pixel width.
        tree = KDTree(λ[(~mask) * (ivar > 0)].reshape((-1, 1)))
        distances, indices = tree.query(λ_out.reshape((-1, 1)), k=1)

        no_data = distances[:-1, 0] > np.diff(λ_out)
        no_data = np.hstack([no_data, no_data[-1]])
        meta["no_data_mask"] = no_data
        if np.any(no_data):
            y_star[no_data] = np.nan
            C_inv_star[no_data] = 0
    
    flags_star = combine_flags(λ_out, λ, flags)

    return (y_star, C_inv_star, flags_star, meta)




class FrizzleOperator(LinearOperator):

    def __init__(self, x: npt.ArrayLike, n_modes: int, **kwargs):
        """
        A linear operator to fit a model to real-valued 1D signals with sine and cosine functions
        using the Flatiron Institute Non-Uniform Fast Fourier Transform.

        :param x:
            The x-coordinates of the data. This should be within the domain (0, 2π).

        :param n_modes:
            The number of Fourier modes to use.

        :param kwargs: [Optional]
            Keyword arguments are passed to the `finufft.Plan()` constructor. 
            Note that the mode ordering keyword `modeord` cannot be supplied.
        """
        if x.dtype == np.float64:
            self.DTYPE_REAL, self.DTYPE_COMPLEX = (np.float64, np.complex128)
        else:
            self.DTYPE_REAL, self.DTYPE_COMPLEX = (np.float32, np.complex64)
        super().__init__(dtype=x.dtype, shape=(len(x), n_modes))
        self.explicit = False
        self.finufft_kwds = dict(dtype=self.DTYPE_COMPLEX.__name__, n_modes_or_dim=(n_modes, ), modeord=0, **kwargs)
        self._plan_matvec = finufft.Plan(2, **self.finufft_kwds)
        self._plan_rmatvec = finufft.Plan(1, **self.finufft_kwds)
        self._plan_matvec.setpts(x)
        self._plan_rmatvec.setpts(x)
        self._hx = n_modes // 2 
        return None
    
    def _pre_process_matvec(self, c):
        return np.hstack([-1j * c[:self._hx], c[self._hx:]], dtype=self.DTYPE_COMPLEX)

    def _post_process_rmatvec(self, f):
        return np.hstack([-f[:self._hx].imag, f[self._hx:].real], dtype=self.DTYPE_REAL)

    def _matvec(self, c):
        return self._plan_matvec.execute(self._pre_process_matvec(c)).real

    def _rmatvec(self, f):
        return self._post_process_rmatvec(self._plan_rmatvec.execute(f.astype(self.DTYPE_COMPLEX)))


def ensure_dict(d, **defaults):
    kwds = dict()
    kwds.update(defaults)
    kwds.update(d or {})
    return kwds

def check_inputs(λ_out, λ, flux, ivar, mask):
    λ, flux = map(np.hstack, (λ, flux))
    if mask is None:
        mask = np.zeros(flux.size, dtype=bool)
    else:
        mask = np.hstack(mask).astype(bool)
    
    λ_out = np.array(λ_out)
    # Mask things outside of the resampling range
    mask *= ((λ_out[0] <= λ) * (λ <= λ_out[-1]))

    if ivar is None:
        ivar = np.ones_like(flux)
    else:
        ivar = np.hstack(ivar)    
    return (λ_out, λ, flux, ivar, mask)

def separate_flags(flags: Optional[npt.ArrayLike] = None):
    """
    Separate flags into a dictionary of flags for each bit.
    
    :param flags:
        An ``M``-length array of flag values.
    
    :returns:
        A dictionary of flags, where each key is a bit and each value is an array of 0s and 1s.
    """
    separated = {}
    if flags is not None:
        for q in range(1 + int(np.log2(np.max(flags)))):
            is_set = (flags & (2**q)) > 0
            if any(is_set):
                separated[q] = is_set.astype(bool)
    return separated    

def combine_flags(λ_out, λ, flags):
    """
    Combine flags from input spectra.

    :param λ_out:
        The wavelengths to sample the combined spectrum on.
    
    :param λ:
        The input wavelengths.
    
    :param flags:
        An array of integer flags.
    """
    flags_star = np.zeros(λ_out.size, dtype=np.uint64 if flags is None else flags.dtype)
    λ_out_T = λ_out.reshape((-1, 1))
    diff_λ_out = np.diff(λ_out)
    for bit, flag in separate_flags(flags).items():
        tree = KDTree(λ[flag].reshape((-1, 1)))            
        distances, indices = tree.query(λ_out_T, k=1)
        within_pixel = np.hstack([distances[:-1, 0] <= diff_λ_out, False])
        flags_star[within_pixel] += 2**bit
    return flags_star