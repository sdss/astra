import numpy as np
import os
import pickle
import warnings
from sklearn.decomposition import NMF
from sklearn.decomposition._nmf import non_negative_factorization
from sklearn.exceptions import ConvergenceWarning
from astropy.nddata import InverseVariance
from astra.tools.continuum.base import Continuum, _pixel_slice_and_mask
from astra.tools.continuum.sinusoids import Sinusoids
from astra.tools.spectrum import SpectralAxis, Spectrum1D
from astra.utils import log, expand_path
from typing import Optional, Union, Tuple, List

from scipy import optimize as op


class Emulator:

    def __init__(
        self,
        components: np.ndarray,
        deg: Optional[int] = 3,
        L: Optional[float] = 1400,
        scalar: Optional[float] = 1e-6,
        spectral_axis: Optional[SpectralAxis] = None,
        regions: Optional[List[Tuple[float, float]]] = None,
        mask: Optional[np.array] = None,
        fill_value: Optional[Union[int, float]] = np.nan,
        expectation_kwds: Optional[dict] = None,
        **kwargs,
    ) -> None:
        
        self.continuum_model = Sinusoids(
            spectral_axis=spectral_axis,
            regions=regions,
            mask=mask,
            fill_value=fill_value,
            deg=deg,
            L=L,
            scalar=scalar
        )
        self.components = components
        self.phi_size = components.shape[0]
        self.theta_size = self.continuum_model.num_regions * (2 * self.continuum_model.deg + 1)
        self._expectation_kwds = expectation_kwds or {}
        return None

    

    def _check_data(self, spectrum):
        N, P = self.continuum_model._get_shape(spectrum)
        flux = spectrum.flux.value.reshape((N, P)).copy()
        ivar = spectrum.uncertainty.represent_as(InverseVariance).array.reshape((N, P)).copy()

        bad_pixels = ~np.isfinite(ivar) | ~np.isfinite(flux) | (ivar == 0)
        flux[bad_pixels] = 0
        ivar[bad_pixels] = 0

        return (flux, ivar)
        

    def _maximization(self, flux, ivar, continuum_args):
        theta = self.continuum_model._fit(flux, ivar, continuum_args)
        continuum = self.continuum_model._evaluate(theta, continuum_args)
        return (theta, continuum)
    

    def _expectation(
            self, 
            flux, 
            W=None,
            solver="mu",
            **kwargs
        ):

        X = 1 - flux # absorption

        # Only use non-negative finite pixels.
        use = np.isfinite(flux) & (X >= 0)
        n_components, n_pixels = self.components.shape
        assert flux.size == n_pixels

        if np.sum(use) < n_components:
            log.warning(f"Number of non-negative finite pixels ({np.sum(use)}) is less than the number of components ({n_components}).")

        kwds = dict(
            # If W is None it means it's the first iteration.
            init=None if W is None else "custom",
            # The documentation says that custom matrices W and H can only be used if `update_H=True`.
            # Since we want it to use W from the previous iteration, we will set `update_H=True`, and ignore H_adjusted.
            update_H=True,
            solver=solver,
            W=W,
            H=self.components[:, use],
            n_components=n_components,
            beta_loss="frobenius",
            tol=1e-4, # Irrelevant since we are only taking 1 step, usually.
            max_iter=1,
            # No regularization! We are at the test step here, and we want our absorption model to be flexible.
            alpha_W=0.0, 
            alpha_H=0.0,
            l1_ratio=1.0,
            random_state=None,
            verbose=0,
            shuffle=False
        )
        # Only include kwargs that non_negative_factorization accepts.
        kwds.update({k: v for k, v in kwargs.items() if k in kwds})

        W_next, H_adjusted, n_iter = non_negative_factorization(X[use].reshape((1, -1)), **kwds)
        rectified_model_flux = 1 - (W_next @ self.components)[0]
        return (W_next, rectified_model_flux, np.sum(use))


    def fit(
            self, 
            spectrum: Spectrum1D, 
            tol: float = 1e-1, 
            max_iter: int = 1_000, 
        ):
        """
        Simultaneously fit the continuum and stellar absorption.

        :param spectrum:
            The spectrum to fit continuum to. This can be multiple visits of the same spectrum.
        
        :param tol: [optional]
            The difference in \chi-squared value between iterations to establish convergence.

            What makes a good tolerance value? There are two parts that contribute to this
            tolerance: the stellar absorption model, and the continuum model. The continuum
            model is linear algebra, so it contributes very little to the tolerance between
            successive iterations. The stellar absorption model is a non-negative matrix
            factorization. This tolerance definitely should not be set to be smaller than
            the tolerance specified when building the non-negative matrix factorization (1e-4),
            because the stellar absorption model has no flexibility to predict absorption
            better than that average degree. For this reason, it's probably sufficient to
            set the tolerance a few orders of magnitude larger (1e-1 or 1e-2), with some
            sensible number of max iterations.

        :param max_iter: [optional]
            The maximum number of iterations.
        
        :returns:
            A tuple of (phi, theta, continuum, model_rectified_flux, meta) where:

            - `phi` are the amplitudes for the non-negative matrix factorization (e.g., `W`)
            - `theta` is the parameters of the continuum model
            - `continuum` is the continuum model evaluated at the spectral axis
            - `model_rectified_flux` is the rectified flux evaluated at the spectral axis
            - `meta` is a dictionary of metadata
        """
        try:
            return self._fit(spectrum, tol, max_iter)
        except:
            raise 
            N, P = self.continuum_model._get_shape(spectrum)
            phi = np.zeros(self.components.shape[0])
            theta = np.zeros((N, self.theta_size))
            continuum = np.ones((N, P)) * np.nan
            model_rectified_flux = np.ones(P) * np.nan
        
            meta = dict(
                chi_sqs=[999],
                reduced_chi_sqs=[999],
                n_pixels_used_in_nmf=0,
                success=False,
                iter=1000,
                message="Failed to fit.",
                continuum_args=None
            )

            return (phi, theta, continuum, model_rectified_flux, meta)

    def _expectation_maximization(self, flux, ivar, stacked_flux, phi, continuum_args, **kwargs):
        phi_next, model_rectified_flux, n_pixels = self._expectation(
            stacked_flux, 
            W=phi.copy() if phi is not None else phi, # make sure you copy 
            **{**self._expectation_kwds, **kwargs}
        )

        theta_next, continuum = self._maximization(
            flux / model_rectified_flux,
            model_rectified_flux * ivar * model_rectified_flux,
            continuum_args
        )        

        chi_sq = ((flux - model_rectified_flux * continuum)**2 * ivar)
        finite = np.isfinite(chi_sq)
        chi_sq = np.sum(chi_sq[finite])

        args = (phi_next, theta_next, continuum, model_rectified_flux, n_pixels, np.sum(finite))
        return (chi_sq, args)


    def _fit(self, spectrum, tol, max_iter):
        flux, ivar = self._check_data(spectrum)
        continuum_args = self.continuum_model._initialize(spectrum)

        with warnings.catch_warnings():
            for category in (RuntimeWarning, ConvergenceWarning):
                warnings.filterwarnings("ignore", category=category)

            phi = None # phi is the same as W used in NMF
            theta, continuum = self._maximization(flux, ivar, continuum_args)

            alpha_Ws = [0]
            min_log_alpha_W, max_log_alpha_W = (None, None)
            if min_log_alpha_W is not None and max_log_alpha_W is not None:
                alpha_Ws = np.hstack([alpha_Ws, np.logspace(min_log_alpha_W, max_log_alpha_W, 1 + max_log_alpha_W - min_log_alpha_W)])
            
            chi_sqs, n_pixels_used_in_chisq, n_pixels_used_in_nmf, alpha_Ws_used = ([], [], [], [])
            for iter in range(max_iter):

                conditional_flux = flux / continuum
                conditional_ivar = continuum * flux * continuum
                stacked_flux = np.sum(conditional_flux * conditional_ivar, axis=0) / np.sum(conditional_ivar, axis=0) 

                # Use increasing regularisation for noisy spectra to ensure we converge.
                for alpha_W in alpha_Ws:
                    chi_sq, em_args = self._expectation_maximization(
                        flux, 
                        ivar, 
                        stacked_flux, 
                        phi, 
                        continuum_args,
                        alpha_W=alpha_W
                    )
                    if iter == 0 or chi_sq < chi_sqs[-1]:
                        # first iteration, or chi-sq improved.
                        break
                else:
                    success, message = (False, "Failed to improve \chi^2")
                    break

                (phi, theta, continuum, model_rectified_flux, n_pixels, n_finite) = em_args

                chi_sqs.append(chi_sq)
                n_pixels_used_in_nmf.append(n_pixels)
                n_pixels_used_in_chisq.append(n_finite)
                alpha_Ws_used.append(alpha_W)
                if alpha_W > 0:
                    print(f"{iter} alpha_W={alpha_W}")

            
                if iter > 0:
                    delta_chi_sq = chi_sqs[-1] - chi_sqs[-2]
                    if (delta_chi_sq < 0) and abs(delta_chi_sq) <= tol:
                        # Converged
                        success, message = (True, f"Convergence reached after {iter} iterations")
                        break
            
            else:
                success, message = (True, f"Convergence not reached after {max_iter} iterations ({abs(delta_chi_sq)} > {tol:.2e})")
                warnings.warn(message)

        reduced_chi_sqs = np.array(chi_sqs) / (np.array(n_pixels_used_in_chisq) - phi.size - theta.size - 1)
        meta = dict(
            chi_sqs=chi_sqs,
            reduced_chi_sqs=reduced_chi_sqs,
            n_pixels_used_in_nmf=n_pixels_used_in_nmf,
            success=success,
            iter=iter,
            message=message,
            continuum_args=continuum_args
        )
        return (phi, theta, continuum, model_rectified_flux, meta)