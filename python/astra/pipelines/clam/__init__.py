
import os
keys = "OPENBLAS_NUM_THREADS MKL_NUM_THREADS OMP_NUM_THREADS NUMBA_NUM_THREADS".split()
for key in keys:
    os.environ[key] = "1"

from peewee import ModelSelect
import pickle
import concurrent.futures
import numpy as np
from astra import __version__, task
from astra.utils import expand_path
from astra.models import ApogeeCoaddedSpectrumInApStar, Source
#from scipy.interpolate import RegularGridInterpolator 
from tqdm import tqdm
from scipy import optimize as op
from scipy.ndimage import map_coordinates
from scipy.spatial import distance
from scipy.linalg import lu_factor, lu_solve
import warnings
from typing import Optional, Iterable, Sequence, Union

from peewee import JOIN
from astra.models.clam import Clam


# my CustomGridInterpolator is about 5x faster than RGI given slinear
from astra.pipelines.clam.rgi import CustomGridInterpolator as RegularGridInterpolator

regions = (
    (15120.0, 15820.0),
    (15840.0, 16440.0),
    (16450.0, 16960.0),
)
modes = 7

def region_slices(λ, regions):
    slices = []
    for region in np.array(regions):
        si, ei = λ.searchsorted(region)
        slices.append(slice(si, ei + 1))
    return slices    


def design_matrix(λ: np.array, modes: int) -> np.array:
    #L = 1300.0
    #scale = 2 * (np.pi / (2 * np.ptp(λ)))
    scale = np.pi / np.ptp(λ)
    return np.vstack(
        [
            np.ones_like(λ).reshape((1, -1)),
            np.array(
                [
                    [np.cos(o * scale * λ), np.sin(o * scale * λ)]
                    for o in range(1, (modes - 1) // 2 + 1)
                ]
            ).reshape((modes - 1, λ.size)),
        ]
    ).T

LARGE = 1e10


def get_next_slice(index: Sequence[float], step: Sequence[float], shape: Sequence[float]):
    """
    Get the next slice given the current position, step, and dimension shapes.
    
    :param index:
        The best current index.
    
    :param step:
        The step size in each dimension.
    
    :param shape:
        The shape of the grid.
    
    :returns:
        A tuple of slices for the next iteration.
    """
    next_slice = []
    for center, step, end in zip(index, step, shape):
        start = center - step
        stop = center + step + 1                
        if start < 0:
            offset = -start
            start += offset
            stop += offset
        elif stop > end:
            offset = stop - (end + 1)
            start -= offset
            stop -= offset
        next_slice.append(slice(start, stop, step))
    return tuple(next_slice)


def initial_guess(
    grid_points,
    A,
    W,
    H,
    log_flux: Sequence[float],
    log_ivar: Sequence[float],
    fixed: Optional[Sequence[Union[None, int]]] = None,
    max_iter: Optional[int] = 10,
    max_step_size: Optional[Union[int, Sequence[int]]] = 8,
    full_output: Optional[bool] = False,
    verbose: Optional[bool] = False,        
) -> Sequence[float]:
    """
    Step through the grid of stellar parameters to find a good initial guess.
    
    :param flux:
        The observed flux.
    
    :param ivar:
        The inverse variance of the observed flux.
    
    :param max_iter: [optional]
        The maximum number of iterations to take.
        
    :param max_step_size: [optional]
        The maximum step size to allow, either as a single value per label dimension, or as a
        tuple of integers. If `None`, then no restriction is made on the maximum possible step,
        which means that in the first iteration the only sampled values in one label might be
        the: smallest value, highest value, and the middle value. Whereas if given, this sets
        the maximum possible step size per dimension. This adds computational cost, but can
        be useful to avoid getting stuck in local minima.
                        
    :param full_output: [optional]
        Return a two-length tuple containing the initial guess and a dictionary of metadata.
        
    :param verbose: [optional]
        Print verbose output.
    
    :returns:
        The initial guess of the model parameters. If `full_output` is true, then an
        additional dictionary of metadata is returned.
    """
    LARGE = 1e11

    not_bad_pixel = ~((~np.isfinite(log_flux) | (log_ivar == 0) | (~np.isfinite(log_ivar))))
    
    
    ATCinv = A[not_bad_pixel].T * log_ivar[not_bad_pixel]
    lu, piv = lu_factor(ATCinv @ A[not_bad_pixel])

    full_shape = W.shape[:-1] # The -1 is for the number of components.        
    rta_indices = tuple(map(np.arange, full_shape))
    
    x = np.empty((*full_shape, A.shape[1]))
    chi2 = LARGE * np.ones(full_shape)
    n_evaluations = 0
    
    if max_step_size is None:
        max_step_size = full_shape

    # Even though this does not get us to the final edge point in some parameters,
    # NumPy slicing creates a *view* instead of a copy, so it is more efficient.                
    current_step = np.clip(
        (np.array(full_shape) - 1) // 2,
        0,
        max_step_size
    )
    if verbose:
        print(f"initial step: {current_step}")
    
    current_slice = list(tuple(slice(0, 1 + end, step) for end, step in zip(full_shape, current_step)))
    if fixed is not None:
        for k, v in enumerate(fixed):
            if isinstance(v, int):
                # slice on this single entry
                current_slice[k] = slice(v, full_shape[k], full_shape[k]) # to keep array sizes, etc
                current_step[k] = 1

    def to_absolute_index(rta, uri):
        uai = []
        for j, v in enumerate(uri):
            try:
                uai.append(rta[j][v])
            except IndexError:
                uai.append(rta[j])
        return tuple(uai)


    current_slice = tuple(current_slice)            
    for n_iter in range(1, 1 + max_iter):      
        W_slice = W[current_slice]
        
        rectified = (W_slice @ -H).reshape((-1, 8575))
        
        # map relative indices to absolute ones
        rta = [rtai[ss] for rtai, ss in zip(rta_indices, current_slice)]
        
        shape = W_slice.shape[:-1]
        for i, r in enumerate(rectified):
            uri = np.unravel_index(i, shape)
            #uai = tuple([(rta[j] if isinstance(rta[j], int) else rta[j][_]) for j, _ in enumerate(uri)])
            uai = to_absolute_index(rta, uri)
            if chi2[uai] < LARGE:
                # We have computed this solution already.
                continue
            x[uai] = lu_solve((lu, piv), ATCinv @ (log_flux[not_bad_pixel] - r[not_bad_pixel]))
            chi2[uai] = np.sum(((A[not_bad_pixel] @ x[uai] + r[not_bad_pixel]) - log_flux[not_bad_pixel])**2 * log_ivar[not_bad_pixel])
            n_evaluations += 1
                        
        # Get next slice
        relative_index = np.unravel_index(np.argmin(chi2[current_slice]), shape)
        #absolute_index = tuple([(rta[j] if isinstance(rta[j], int) else rta[j][_]) for j, _ in enumerate(relative_index)])
        absolute_index = to_absolute_index(rta, relative_index)
        
        if verbose:
            print("current_slice: ", current_slice)
            #for i, cs in enumerate(current_slice):
            #    print(f"\t{self.label_names[i]}: {self.grid_points[i][cs]}")
            #print(f"n_iter={n_iter}, chi2={chi2[absolute_index]}, x={[p[i] for p, i in zip(self.grid_points, absolute_index)]}")
            #print(f"absolute index {absolute_index} -> {dict(zip(self.label_names, [p[i] for p, i in zip(self.grid_points, absolute_index)]))}")
            
        next_step = np.clip(
            np.clip(current_step // 2, 1, full_shape),
            0,
            max_step_size
        )    
        next_slice = list(get_next_slice(absolute_index, next_step, full_shape))
        if fixed:
            for k, v in enumerate(fixed):
                if isinstance(v, int):
                    next_step[k] = 1
                    next_slice[k] = slice(v, full_shape[k], full_shape[k])
        next_slice = tuple(next_slice)
                    
        if verbose:
            print("next_slice", next_slice)

        if next_slice == current_slice and max(next_step) == 1:
            if verbose:
                print("stopping")
            break
        
        current_step, current_slice = (next_step, next_slice)
    else:
        warnings.warn(f"Maximum iterations reached ({max_iter}) for initial guess")

    
    upper = np.array(list(map(np.max, grid_points)))
    lower = np.array(list(map(np.min, grid_points)))
    p_init = np.hstack([
        [p[i] for p, i in zip(grid_points, absolute_index)],
        x[absolute_index]
    ])    

    import matplotlib.pyplot as plt

    

    raise a

    raise a



  
    return p_init
    


@task
def clam(
    spectra = (
        ApogeeCoaddedSpectrumInApStar
        .select()
        .where(ApogeeCoaddedSpectrumInApStar.spectrum_pk == 2981929)
        #.join(Source)
        #.switch(ApogeeCoaddedSpectrumInApStar)
        #.join(Clam, JOIN.LEFT_OUTER, on=(Clam.spectrum_pk == ApogeeCoaddedSpectrumInApStar.spectrum_pk))
        #.where(Source.sdss4_apogee_member_flags > 0)
        #.where(Clam.spectrum_pk.is_null())
        #.where(Source.flag_sdss4_apogee_member_pleiades)
    ),
    page=None,
    limit=None,
    modes=7,
    regions = (
        (15120.0, 15820.0),
        (15840.0, 16440.0),
        (16450.0, 16960.0),
    )
) -> Iterable[Clam]:

    if isinstance(spectra, ModelSelect) and limit is not None:
        if page is not None:
            spectra = spectra.paginate(page, limit)
        else:
            spectra = spectra.limit(limit)

    
    λ = 10**(4.179 + 6e-6 * np.arange(8575))
    A_continuum = np.zeros((λ.size, len(regions) * modes), dtype=float)
    for i, region_slice in enumerate(region_slices(λ, regions)):
        A_continuum[region_slice, i*modes:(i+1)*modes] = design_matrix(λ[region_slice], modes)

    #with open(expand_path("~/sas_home/continuum/20240806.pkl"), "rb") as fp:
    with open(expand_path("~/20240807.pkl"), "rb") as fp:
        label_names, grid_points, W, H_clipped, model_scatter = pickle.load(fp)

    if W.ndim == 2:
        shape = tuple(map(len, grid_points))
        W = W.reshape((*shape, -1))

    model_var = np.inf * np.ones(8575)
    #model_var[:50] = model_scatter[-50:] = np.inf
    model_var[50:-50] = model_scatter**2

    label_names = ("v_sini", "n_m", "c_m", "m_h", "v_micro", "logg", "teff")

    '''
    is_warm = (grid_points[-1] >= 4000)
    rgi_warm = RegularGridInterpolator(
        grid_points[:-1] + [grid_points[-1][is_warm]],
        W[..., is_warm, :],
        #method="slinear",
        #bounds_error=False,
        #fill_value=0.0 
    )
    rgi_cool = RegularGridInterpolator(
        grid_points[:-1] + [grid_points[-1][~is_warm]],
        W[..., ~is_warm, :],
        #method="slinear",
        #bounds_error=False,
        #fill_value=0.0
    )

    def rgi(θ, **kwargs):
        if θ[-1] >= 4000:
            _rgi = rgi_warm
        else:
            _rgi = rgi_cool        
        # Calling ._spline directly does not protect us from extrapolation.
        #v = _rgi._spline(θ, **kwargs)
        v = _rgi(θ, **kwargs)
        v[~np.isfinite(v)] = 0
        return v
        #return _rgi(θ, **kwargs).flatten()    
    '''

    rgi = RegularGridInterpolator(grid_points, W)

    n_components = H_clipped.shape[0]
    W = W.reshape((-1, n_components))

    H = np.zeros((H_clipped.shape[0], 8575))
    H[:, 50:-50] = H_clipped

    n_components, n_pixels = H.shape
    n_labels = len(label_names)
    A = np.vstack([-H, A_continuum.T]).T

    bounds = np.zeros((2, n_labels + A_continuum.shape[1]))
    bounds[0] = -np.inf
    bounds[1] = +np.inf
    bounds[0, :n_labels] = list(map(np.min, grid_points))
    bounds[1, :n_labels] = list(map(np.max, grid_points))

    shape = tuple(map(len, grid_points))

    max_iter = int(1e20)
    epsilon = np.finfo(float).eps

    lsq_bounds = np.zeros((2, A.shape[1]))
    lsq_bounds[0] = -np.inf
    lsq_bounds[1] = +np.inf
    lsq_bounds[0, :n_components] = 0
    lsq_bounds[1, :n_components] = np.max(W, axis=0)

    cov = np.zeros((A.shape[1], A.shape[1]))
    cov[:n_components, :n_components] = np.cov(W.T)

    cov *= 1e11 # MAGIC

    '''
    def get_rgi(θ, **kwargs):
        if θ[-1] >= 4000:
            _rgi = rgi_warm
        else:
            _rgi = rgi_cool        
        return _rgi
    '''

    W_full = W.reshape((*tuple(map(len, grid_points)), -1))

    def _fit_spectrum(spectrum):
        try:
            z = np.log(spectrum.flux)
        except:
            return dict(
                spectrum_pk=spectrum.spectrum_pk,
                source_pk=spectrum.source_pk,
                flag_spectrum_io_error=True
            )

        z_ivar = spectrum.ivar * spectrum.flux**2
        z_sigma = np.sqrt(1/z_ivar + model_var)
        use = (spectrum.ivar > 0) * np.isfinite(z) * np.isfinite(z_sigma)

        Au = A[use]
        nHu = -H[:, use]
        AcuT = A_continuum[use].T

        p0 = initial_guess(
            grid_points,
            A_continuum, 
            W_full, 
            H, 
            z, 
            z_ivar, 
            fixed=(None, 2, 2, None, None, None, None),
            verbose=False
        )
        p0_labels = p0[:n_labels]

        '''
        ATCinv = Au.T * z_ivar[use]
        
        # switching from `epsilon` to 1e-8 made this ... somehow slower?!
        r_lsq = op.lsq_linear(
            ATCinv @ Au + cov,
            ATCinv @ z[use],
            bounds=lsq_bounds,
            method="bvls",
            lsq_solver="exact",
            tol=epsilon,
            lsmr_tol=epsilon,
            max_iter=max_iter,
            lsmr_maxiter=max_iter        
        )
        dist = distance.cdist(W, r_lsq.x[:n_components].reshape((1, -1)))
        v = np.unravel_index(np.argmin(dist), shape)
        p0_labels = np.array([gp[k] for gp, k in zip(grid_points, v)])
        p0_continuum = r_lsq.x[n_components:]
        # TODO: Update our continuum estimate conditioned at this grid point?
        p0 = np.hstack([p0_labels, p0_continuum])
        '''


        def forward_model(x, *θ):
            #rgi = get_rgi(θ[:n_labels])
            X = np.hstack([rgi(θ[:n_labels]).flatten(), θ[n_labels:]])
            return Au @ X

        def jac_forward_model(x, *θ):    
            #dθ_dW = get_rgi(θ[:n_labels]).jacobian(θ[:n_labels])
            dθ_dW = rgi.jacobian(θ[:n_labels])
            return np.vstack([
                (dθ_dW @ nHu),
                AcuT
            ]).T            

        try:
            μ, Σ = op.curve_fit(
                forward_model,
                xdata=None,
                ydata=z[use],
                sigma=z_sigma[use],
                p0=p0,
                bounds=bounds,
                jac=jac_forward_model,
                check_finite=False,
            )
        except:
            print(f"Failed on spectrum {spectrum}")
            return dict(
                spectrum_pk=spectrum.spectrum_pk,
                source_pk=spectrum.source_pk,
                flag_runtime_error=True
            )


        rectified_flux = np.exp(rgi(μ[:n_labels]) @ -H).flatten()
        continuum = np.exp(A_continuum @ μ[n_labels:])
        chi2 = (spectrum.flux - (continuum * rectified_flux))**2 * spectrum.ivar
        rchi2 = np.sum(chi2[use]) / (np.sum(use) - len(μ) - 1)

        initial_labels = p0_labels
        initial_labels[0] = 10**initial_labels[0]

        labels = μ[:n_labels]
        labels[0] = 10**labels[0]
        
        e_labels = np.sqrt(np.diag(Σ[:n_labels, :n_labels]))
        e_labels[0] = labels[0] * e_labels[0] * np.log(10) # 10^y * e_y * ln(10)

        result = dict(zip(label_names, labels)) # labels
        result.update(dict(zip([f"e_{ln}" for ln in label_names], e_labels))) # errors
        result.update(dict(zip([f"initial_{ln}" for ln in label_names], initial_labels))) # initial labels
        result.update(
            rchi2=rchi2, 
            theta=μ[n_labels:],
            spectrum_pk=spectrum.spectrum_pk,
            source_pk=spectrum.source_pk
        )

        return result

    for spectrum in tqdm(spectra):
        yield Clam(**_fit_spectrum(spectrum))

    raise a
    with concurrent.futures.ThreadPoolExecutor(128) as executor:
        futures = [executor.submit(_fit_spectrum, spectrum) for spectrum in spectra]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            yield Clam(**future.result())
        

    #for spectrum in tqdm(spectra):
    #result = _fit_spectrum(spectrum)
    #e    print(result)
    if False:

        '''
        t_init = time()
        z_ivar = spectrum.ivar * spectrum.flux**2
        z_sigma = np.sqrt(1/z_ivar + model_var)
        use = (spectrum.ivar > 0) * np.isfinite(z) * np.isfinite(z_sigma)
        t_data_prep = time() - t_init



        Au = A[use]
        nHu = -H[:, use]
        AcuT = A_continuum[use].T

        t_init = time()
        def forward_model(x, *θ):
            X = np.hstack([rgi(θ[:n_labels]).flatten(), θ[n_labels:]])
            return Au @ X

        def jac_forward_model(x, *θ):    
            if θ[n_labels - 1] >= 4000:
                _rgi = rgi_warm
            else:
                _rgi = rgi_cool        
            dθ_dW = _rgi.jacobian(θ[:n_labels])
            return np.vstack([
                (dθ_dW @ nHu),
                AcuT
            ]).T
            
        t_define_functions = time() - t_init

        t_init = time()
        ATCinv = Au.T * z_ivar[use]
        
        # switching from `epsilon` to 1e-8 made this ... somehow slower?!
        r_lsq = op.lsq_linear(
            ATCinv @ Au + cov,
            ATCinv @ z[use],
            bounds=lsq_bounds,
            method="bvls",
            lsq_solver="exact",
            tol=epsilon,
            lsmr_tol=epsilon,
            max_iter=max_iter,
            lsmr_maxiter=max_iter        
        )
        t_lsq = time() - t_init

        # Get nearest grid point
        t_init = time()
        dist = distance.cdist(W, r_lsq.x[:n_components].reshape((1, -1)))
        v = np.unravel_index(np.argmin(dist), W.shape[:-1])
        p0_labels = np.array([gp[k] for gp, k in zip(grid_points, v)])
        t_dist = time() - t_init
        
        p0_continuum = r_lsq.x[n_components:]
        # TODO: Update our continuum estimate conditioned at this grid point?
        p0 = np.hstack([p0_labels, p0_continuum])
        
        t_init = time()
        try:
            μ, Σ = op.curve_fit(
                forward_model,
                xdata=None,
                ydata=z[use],
                sigma=z_sigma[use],
                p0=p0,
                bounds=bounds,
                jac=jac_forward_model,
                check_finite=False,
            )
        except:
            print(f"Failed on spectrum {spectrum}")
            yield Clam(
                spectrum_pk=spectrum.spectrum_pk,
                source_pk=spectrum.source_pk,
                flag_runtime_error=True
            )
            continue

        t_curve_fit = time() - t_init

        t_init = time()
        rectified_flux = np.exp(rgi(μ[:n_labels]) @ -H).flatten()
        continuum = np.exp(A_continuum @ μ[n_labels:])
        chi2 = (spectrum.flux - (continuum * rectified_flux))**2 * spectrum.ivar
        rchi2 = np.sum(chi2[use]) / (np.sum(use) - len(μ) - 1)

        initial_labels = p0_labels
        initial_labels[0] = 10**initial_labels[0]

        labels = μ[:n_labels]
        labels[0] = 10**labels[0]
        
        e_labels = np.sqrt(np.diag(Σ[:n_labels, :n_labels]))
        e_labels[0] = labels[0] * e_labels[0] * np.log(10) # 10^y * e_y * ln(10)

        result = dict(zip(label_names, labels)) # labels
        print(result)
        result.update(dict(zip([f"e_{ln}" for ln in label_names], e_labels))) # errors
        result.update(dict(zip([f"initial_{ln}" for ln in label_names], initial_labels))) # initial labels
        result.update(rchi2=rchi2, theta=μ[n_labels:])
        t_meta = time() - t_init

        print(f"data={t_data_prep:.1e} define={t_define_functions:.1e}, lsq={t_lsq:.1e}, dist={t_dist:.1e}, curve={t_curve_fit:.1e}, meta={t_meta:.1e}")

        yield Clam(
            spectrum_pk=spectrum.spectrum_pk, 
            source_pk=spectrum.source_pk,
            **result
        )
        '''



#def do_clam(nodes, procs):


