import os
import numpy as np
import pickle
import warnings
from itertools import cycle
from functools import cached_property
from scipy import optimize as op
from sklearn.linear_model import Lasso, LinearRegression
from joblib import Parallel, delayed
from time import time
from tqdm import tqdm

from sklearn.exceptions import ConvergenceWarning

from astra import log
from astra.utils import expand_path

class CannonModel:
    
    """
    A second-order polynomial Cannon model.
    
    For example, the generative model for two labels (teff, logg) might look something
    like:
    
        f(\theta) = \theta_0 
                  + (\theta_1 * teff) 
                  + (\theta_2 * teff^2)
                  + (\theta_3 * logg)
                  + (\theta_4 * logg * teff)
                  + (\theta_5 * logg^2)
    """
    
    def __init__(
        self, 
        training_labels,
        training_flux,
        training_ivar,
        label_names,
        dispersion=None,
        regularization=0,
        n_threads=-1,
        **kwargs
    ) -> None:
        self.training_labels, self.training_flux, self.training_ivar, self.offsets, self.scales \
            = _check_inputs(label_names, training_labels, training_flux, training_ivar, **kwargs)
        self.label_names = label_names
        self.dispersion = dispersion
        self.regularization = regularization
        self.n_threads = n_threads or 1
        if self.n_threads < 0:
            self.n_threads = os.cpu_count()

        # If we are loading from a pre-trained model.
        self.theta = kwargs.get("theta", None)
        self.s2 = kwargs.get("s2", None)
        self.meta = kwargs.get("meta", {})
        return None

    @property
    def trained(self):
        """ Boolean property defining whether the model is trained. """
        return self.theta is not None and self.s2 is not None
        
    def write(self, path, save_training_set=False, overwrite=False):
        """
        Write the model to disk.
        
        :param path:
            The path to write the model to.
        
        :param save_training_set: [optional]
            Include the training set in the saved model (default: False).
        """
        full_path = expand_path(path)
        if os.path.exists(full_path) and not overwrite:
            raise FileExistsError(f"File {full_path} already exists.")
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        if not save_training_set and not self.trained:
            raise ValueError("Nothing to save: model not trained and save_training_set is False")

        keys = ["theta", "s2", "dispersion", "regularization", "n_threads", "label_names", "meta", "offsets", "scales"]
        if save_training_set:
            keys += ["training_labels", "training_flux", "training_ivar"]
    
        state = { k: getattr(self, k) for k in keys }
        with open(path, "wb") as fp:
            pickle.dump(state, fp)
        log.info(f"Wrote model to {path}")

        return True
    
    
    @classmethod
    def read(cls, path):
        """Read a model from disk."""

        full_path = expand_path(path)
        with open(full_path, "rb") as fp:
            state = pickle.load(fp)
        # if there's no training data, just give Nones
        for k in ("training_labels", "training_flux", "training_ivar"):
            state.setdefault(k, None)
        return cls(**state)

    @property
    def term_descriptions(self):
        """
        Return descriptions for all the terms in the design matrix.
        """
        js, ks = _design_matrix_indices(len(self.label_names))
        terms = []
        for j, k in zip(js, ks):
            if j == 0 and k == 0:
                terms.append(1)
            else:
                term = []
                if j > 0:
                    term.append(self.label_names[j - 1])
                if k > 0:
                    term.append(self.label_names[k - 1])
                terms.append(tuple(term))
        return terms

    @property
    def term_type_indices(self):
        """
        Returns a three-length tuple that contains:
        - indices of linear terms in the design matrix
        - indices of quadratic terms in the design matrix
        - indices of cross-terms in the design matrix
        """
        js, ks = _design_matrix_indices(len(self.label_names))
        indices = [[], [], []]
        for i, (j, k) in enumerate(zip(js, ks)):
            if j == 0 and k == 0: continue

            if min(j, k) == 0 and max(j, k) > 0:
                # linear term
                indices[0].append(i)
            elif j > 0 and j == k:
                # quadratic term
                indices[1].append(i)
            else:
                # cross-term
                indices[2].append(i)
        return indices

    @cached_property
    def _design_matrix_indices(self):
        return _design_matrix_indices(len(self.label_names))

    def train(self, hide_warnings=True, tqdm_kwds=None, **kwargs):
        """
        Train the model.
        
        :param hide_warnings: [optional]
            Hide convergence warnings (default: True). Any convergence warnings will be recorded in
            `model.meta['warnings']`, which can be accessed after training.

        :param tqdm_kwds: [optional]
            Keyword arguments to pass to `tqdm` (default: None).
        """

        # Calculate design matrix without bias term, using normalized labels
        X = _design_matrix(
            _normalize(self.training_labels, self.offsets, self.scales),
            self._design_matrix_indices
        )[:, 1:]
        flux, ivar = self.training_flux, self.training_ivar
        N, L = X.shape
        N, P = flux.shape

        _tqdm_kwds = dict(total=P, desc="Training")
        _tqdm_kwds.update(tqdm_kwds or {})

        args = (X, self.regularization, hide_warnings)
        t_init = time()
        results = Parallel(self.n_threads, prefer="processes")(
            delayed(_fit_pixel)(p, Y, W, *args, **kwargs) 
               for p, (Y, W) in tqdm(enumerate(zip(flux.T, ivar.T)), **_tqdm_kwds)
        )
        t_train = time() - t_init

        self.s2 = np.zeros(P)
        self.theta = np.zeros((1 + L, P))
        self.meta.update(
            t_train=t_train, 
            train_warning=np.zeros(P, dtype=bool),
            n_iter=np.zeros(P, dtype=int),
            dual_gap=np.zeros(P, dtype=float)
        )
        for index, pixel_theta, pixel_s2, meta in results:
            self.s2[index] = pixel_s2
            self.theta[:, index] = pixel_theta
            self.meta["train_warning"][index] = meta.get("warning", False)
            self.meta["n_iter"][index] = meta.get("n_iter", -1)
            self.meta["dual_gap"][index] = meta.get("dual_gap", np.nan)
        return self

    
    def predict(self, labels):
        """
        Predict spectra, given some labels.
        """
        L = _normalize(np.atleast_2d(labels), self.offsets, self.scales)
        return _design_matrix(L, self._design_matrix_indices) @ self.theta


    def chi_sq(self, labels, flux, ivar, aggregate=np.sum):
        """
        Return the total \chi^2 difference of the expected flux given the labels, and the observed
        flux. The total inverse variance (model and observed) is used to weight the \chi^2 value.
        
        :param labels:
            An array of stellar labels with shape `(n_spectra, n_labels)`.
        
        :param flux:
            An array of observed flux values with shape `(n_spectra, n_pixels)`.
        
        :param ivar:
            An array containing the inverse variance of the observed flux, with shape `(n_spectra, n_pixels)`.
        """
        adjusted_ivar = (ivar / (1. + ivar * self.s2))
        return aggregate(adjusted_ivar * (self.predict(labels) - flux)**2)


    def reduced_chi_sq(self, labels, flux, ivar, aggregate=np.sum):
        nu = aggregate(ivar > 0) - labels.size
        return self.chi_sq(labels, flux, ivar, aggregate) / nu




    def fit_spectrum(self, flux, ivar, x0=None, tqdm_kwds=None, n_threads=None, prefer="processes"):
        """
        Return the stellar labels given the observed flux and inverse variance.

        :param flux:
            An array of observed flux values with shape `(n_spectra, n_pixels)`.
        
        :param ivar:
            An array containing the inverse variance of the observed flux, with shape `(n_spectra, n_pixels)`.

        :param x0: [optional]
            An array of initial values for the stellar labels with shape `(n_spectra, n_labels)`. If `None`
            is given (default) then the initial guess will be estimated by linear algebra.

        :param tqdm_kwds: [optional]
            Keyword arguments to pass to `tqdm` (default: None).
        """

        flux, ivar = np.atleast_2d(flux), np.atleast_2d(ivar)
        sigma = (ivar / (1. + ivar * self.s2))**-0.5
        N, P = flux.shape
        L = len(self.label_names)

        _tqdm_kwds = dict(total=N, desc="Fitting")
        _tqdm_kwds.update(tqdm_kwds or {})

        # NOTE: Here we pre-calculate the tril_indices for the design matrix calculation.
        #       This should match exactly what is done elsewhere for the design matrix, or
        #       you're fired!
        args = (self.theta, self._design_matrix_indices, self.offsets, self.scales)
        if x0 is None:
            x0 = _initial_guess(flux, sigma, *args)
        else:
            x0 = np.atleast_2d(x0)
            
        iterable = tqdm(zip(flux, sigma, x0), **_tqdm_kwds)

        n_threads = n_threads or os.cpu_count()
        if N == 1 or n_threads in (1, None):
            results = [_fit_spectrum(*data, *args) for data in iterable]
        else:
            results = Parallel(self.n_threads, prefer=prefer)(
                delayed(_fit_spectrum)(*data, *args) for data in iterable
            )
        return results


BIG = 1
SMALL = 1e-12

def _design_matrix_indices(L):
    return np.tril_indices(1 + L)

def _design_matrix(labels, idx):
    N, L = labels.shape
    #idx = _design_matrix_indices(L)
    iterable = np.hstack([np.ones((N, 1)), labels])[:, np.newaxis]
    return np.vstack([l.T.dot(l)[idx] for l in iterable])
        

def _initial_guess(flux, sigma, theta, idx=None, offsets=0, scales=1, clip_sigma=3):
    use = np.all(np.isfinite(sigma), axis=0)
    # Solve for AX = B where B = flux - theta_0 and A is the design matrix (except bias)  
    A = theta[1:, use].T
    B = (flux[:, use] - theta[0, use]).T
    try:
        X, residuals, rank, singular = np.linalg.lstsq(A, B, rcond=-1)
    except np.linalg.LinAlgError:
        # start at central value
        return offsets
    else:
        if idx is None:
            # Solve for the number of labels we expect (ignore the negative root)
            # The number of items in a lower triangular matrix is X = N(N+1)/2 -> -2(X + 1) + N + N^2 = 0
            # (The X + 1) here arises because we exclude the theta_0 term and solve for Y - \theta_0.
            L = int(np.max(np.polynomial.Polynomial([-2*(X.size + 1), 1, 1]).roots())) - 1
            idx = _design_matrix_indices(L)

        # Need the indices of the linear terms.
        x0 = X[(idx[1] == 0)[1:]].T # offset by 1 to skip missing bias term
        if clip_sigma is not None:
            x0 = np.clip(x0, -clip_sigma, +clip_sigma)

        return _denormalize(x0, offsets, scales)


def _fit_spectrum(flux, sigma, x0, theta, idx, offsets=0, scales=1, clip_sigma=3):
    # NOTE: Here the design matrix is calculated with *DIFFERENT CODE* than what is used
    #       to construct the design matrix during training. The result should be exactly
    #       the same, but here we are taking advantage of not having to call np.tril_indices
    #       with every log likelihood evaluation.
    def f(_, *labels):
        l = np.atleast_2d(np.hstack([1, labels]))
        A = l.T.dot(l)[idx][np.newaxis]
        return (A @ theta)[0]

    L = len(x0)
    try:
        p_opt_norm, cov_norm = op.curve_fit(
            f,
            None,
            flux,
            p0=_normalize(x0, offsets, scales),
            sigma=sigma,
            absolute_sigma=True,
            maxfev=10_000
        )
        model_flux = f(None, *p_opt_norm)
    except:
        N, P = np.atleast_2d(flux).shape
        p_opt = np.nan * np.ones(L)
        cov = np.nan * np.ones((L, L))
        meta = dict(
            chi_sq=np.nan,
            reduced_chi_sq=np.nan,
        )
    else:
        p_opt = _denormalize(p_opt_norm, offsets, scales)
        cov = cov_norm * scales**2 # TODO: define this with _normalize somehow

        # NOTE: Here we are calculating chi-sq with *DIFFERENT CODE* than what is used elsewhere.
        chi_sq = np.sum((flux - model_flux)**2 / sigma**2)
        nu = np.sum(np.isfinite(sigma)) - L
        reduced_chi_sq = chi_sq / nu
        meta = dict(
            chi_sq=chi_sq,
            reduced_chi_sq=reduced_chi_sq,
            p_opt_norm=p_opt_norm
        )
    finally:
        return (p_opt, cov, meta)


def _fit_pixel(index, Y, W, X, alpha, hide_warnings=True, **kwargs):
    N, T = X.shape
    if np.allclose(W, np.zeros_like(W)):
        return (index, np.zeros(1 + T), 0, {})

    if alpha == 0:
        kwds = dict(**kwargs) # defaults
        lm = LinearRegression(**kwds)
    else:
        # defaults:
        kwds = dict(normalize=False, max_iter=10_000, tol=1e-4, precompute=True)
        kwds.update(**kwargs) # defaults
        lm = Lasso(alpha=alpha, **kwds)

    args = (X, Y, W)
    t_init = time()
    if hide_warnings:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            lm.fit(*args)
    else:
        lm.fit(*args)
    t_fit = time() - t_init
    
    theta = np.hstack([lm.intercept_, lm.coef_])
    meta = dict(t_fit=t_fit)
    for attribute in ("n_iter", "dual_gap"):
        try:
            meta[attribute] = getattr(lm, f"{attribute}_")
        except:
            continue

    if "n_iter" in meta:
        meta["warning"] = meta["n_iter"] >= lm.max_iter

    l2 = (Y - lm.predict(X))**2
    mask = W > 0
    inv_W = np.zeros_like(W)
    inv_W[mask] = 1 / W[mask]
    inv_W[~mask] = BIG
    s2 = max(0, np.median(l2 - inv_W, axis=0))

    return (index, theta, s2, meta)


def _check_inputs(label_names, labels, flux, ivar, **kwargs):
    if labels is None and flux is None and ivar is None:
        # Try to get offsets and scales from kwargs
        offsets, scales = (kwargs.get(k, None) for k in ("offsets", "scales"))
        if offsets is None or scales is None:
            log.warning(f"No training set labels given, and no offsets or scales provided!")
            offsets, scales = (0, 1)
        return (labels, flux, ivar, offsets, scales)
    L = len(label_names)
    labels = np.atleast_2d(labels)
    flux = np.atleast_2d(flux)
    ivar = np.atleast_2d(ivar)

    N_0, L_0 = labels.shape
    N_1, P_1 = flux.shape
    N_2, P_2 = ivar.shape

    if L_0 != L:
        raise ValueError(f"{L} label names given but input labels has shape {labels.shape} and should be (n_spectra, n_labels)")

    if N_0 != N_1:
        raise ValueError(
            f"labels should have shape (n_spectra, n_labels) and flux should have shape (n_spectra, n_pixels) "
            f"but labels has shape {labels.shape} and flux has shape {flux.shape}"
        )
    if N_1 != N_2 or P_1 != P_2:
        raise ValueError(
            f"flux and ivar should have shape (n_spectra, n_pixels) "
            f"but flux has shape {flux.shape} and ivar has shape {ivar.shape}"
        )
    
    if L_0 > N_0:
        raise ValueError(f"I don't believe that you have more labels than spectra")

    # Restrict to things that are fully sampled.
    good = np.all(ivar > 0, axis=0)
    ivar = np.copy(ivar)
    ivar[:, ~good] = 0

    # Calculate offsets and scales.
    offsets, scales = _offsets_and_scales(labels)
    if not np.all(np.isfinite(offsets)):
        raise ValueError(f"offsets are not all finite: {offsets}")
    if len(offsets) != L:
        raise ValueError(f"{len(offsets)} offsets given but {L} are needed")
    
    if not np.all(np.isfinite(scales)):
        raise ValueError(f"scales are not all finite: {scales}")
    if len(scales) != L:
        raise ValueError(f"{len(scales)} scales given but {L} are needed")

    return (labels, flux, ivar, offsets, scales)


def _offsets_and_scales(labels):
    return (np.mean(labels, axis=0), np.std(labels, axis=0))
    
def _normalize(labels, offsets, scales):
    return (labels - offsets) / scales

def _denormalize(labels, offsets, scales):
    return labels * scales + offsets
