import numpy as np
import os
import pickle
import warnings
from sklearn.decomposition import NMF
from sklearn.decomposition._nmf import non_negative_factorization, _fit_multiplicative_update
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
        alpha_W: Optional[float] = 1e-5,
        nmf_solver: Optional[str] = "mu",
        nmf_max_iter: Optional[int] = 100,
        nmf_tol: Optional[float] = 1e-1,
        deg: Optional[int] = 3,
        L: Optional[float] = 1400,
        scalar: Optional[float] = 1e-6,
        spectral_axis: Optional[SpectralAxis] = None,
        regions: Optional[List[Tuple[float, float]]] = None,
        mask: Optional[np.array] = None,
        fill_value: Optional[Union[int, float]] = np.nan,
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
        self.mask = mask
        self.alpha_W = alpha_W
        self.nmf_solver = nmf_solver
        self.nmf_max_iter = nmf_max_iter
        self.nmf_tol = nmf_tol
        self.components = components
        self.phi_size = components.shape[0]
        self.theta_size = self.continuum_model.num_regions * (2 * self.continuum_model.deg + 1)
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
    

    def _expectation(self, flux, W, **kwargs):

        absorption = 1 - flux # absorption

        # Only use non-negative finite pixels.
        use = np.isfinite(flux) & (absorption >= 0)
        n_components, n_pixels = self.components.shape
        assert flux.size == n_pixels

        if self.mask is not None:
            use *= ~self.mask

        if np.sum(use) < n_components:
            log.warning(f"Number of non-negative finite pixels ({np.sum(use)}) is less than the number of components ({n_components}).")

        X = absorption[use].reshape((1, -1))
        H = self.components[:, use]


        # TODO: Scale alpha_W based on the number of pixels being used in the mask?    
        kwds = dict(
            # If W is None it means it's the first iteration.
            init=None if W is None else "custom",
            # The documentation says that custom matrices W and H can only be used if `update_H=True`.
            # Since we want it to use W from the previous iteration, we will set `update_H=True`, and ignore H_adjusted.
            update_H=True,
            solver=self.nmf_solver,
            W=W,
            H=H,
            n_components=n_components,
            beta_loss="frobenius",
            tol=self.nmf_tol,
            max_iter=self.nmf_max_iter,
            # Only regularization on W, because we are at the test step here.
            alpha_W=self.alpha_W, 
            alpha_H=0.0,
            l1_ratio=1.0,
            random_state=None,
            verbose=0,
            shuffle=False
        )
        # Only include kwargs that non_negative_factorization accepts.
        kwds.update({k: v for k, v in kwargs.items() if k in kwds})

        W_next, H_adjusted_and_masked, n_iter = non_negative_factorization(X, **kwds)
        #H_adjusted = np.zeros(self.components.shape, dtype=float)
        #H_adjusted[:, use] = H_adjusted_and_masked

        #if adjusted:
        #    use_H = H_adjusted
        #else:
        use_H = self.components

        rectified_model_flux = 1 - (W_next @ use_H)[0]
        return (W_next, rectified_model_flux, np.sum(use), n_iter)
        '''

        X = absorption[use].reshape((1, -1))
        kwds = dict(
            beta_loss="frobenius",
            max_iter=self.nmf_max_iter,
            tol=self.nmf_tol,
            l1_reg_W=self.alpha_W,
            l1_reg_H=0,
            l2_reg_W=0,
            l2_reg_H=0,
            update_H=False,
            verbose=1
        )
        if W is None:
            n_components = H.shape[0]
            avg = np.sqrt(X.mean() / n_components)
            W = np.full((1, n_components), avg, dtype=X.dtype)

        W_next, H_adjusted, n_iter = _fit_multiplicative_update(
            X,
            W=W,
            H=H,
            **kwds
        )
        rectified_model_flux = 1 - (W_next @ self.components)[0]
        return (W_next, rectified_model_flux, np.sum(use), n_iter)
        '''



    def fit(self, spectrum: Spectrum1D, tol: float = 1e-1, max_iter: int = 1_000, initial_phi=None):
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
            The maximum number of expectation-maximization iterations.
        
        :returns:
            A tuple of (phi, theta, continuum, model_rectified_flux, meta) where:

            - `phi` are the amplitudes for the non-negative matrix factorization (e.g., `W`)
            - `theta` is the parameters of the continuum model
            - `continuum` is the continuum model evaluated at the spectral axis
            - `model_rectified_flux` is the rectified flux evaluated at the spectral axis
            - `meta` is a dictionary of metadata
        """
        try:
            return self._fit(spectrum, tol, max_iter, initial_phi=initial_phi)
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
        phi_next, model_rectified_flux, n_pixels, n_nmf_iter = self._expectation(
            stacked_flux, 
            W=phi.copy() if phi is not None else phi, # make sure you copy 
            **kwargs
        )

        theta_next, continuum = self._maximization(
            flux / model_rectified_flux,
            model_rectified_flux * ivar * model_rectified_flux,
            continuum_args
        )        

        chi_sq = ((flux - model_rectified_flux * continuum)**2 * ivar)
        finite = np.isfinite(chi_sq)
        chi_sq = np.sum(chi_sq[finite])

        args = (phi_next, theta_next, continuum, model_rectified_flux, n_pixels, np.sum(finite), n_nmf_iter)
        return (chi_sq, args)


    def _fit(self, spectrum, tol, max_iter, initial_phi=None):
        flux, ivar = self._check_data(spectrum)
        continuum_args = self.continuum_model._initialize(spectrum)

        with warnings.catch_warnings():
            for category in (RuntimeWarning, ConvergenceWarning):
                warnings.filterwarnings("ignore", category=category)

            phi = None
            if initial_phi is not None:
                initial_rectified_flux = 1 - (initial_phi @ self.components)[0]
            else:
                initial_rectified_flux = 1
        
            theta, continuum = self._maximization(
                flux / initial_rectified_flux, 
                initial_rectified_flux * ivar * initial_rectified_flux, 
                continuum_args
            )

            # initial trick
            #continuum *= 1.5
            #print("doing a hack")
            
            chi_sqs, n_pixels_used_in_chisq, n_pixels_used_in_nmf, n_nmf_iters = ([], [], [], [])
            for iter in range(max_iter):

                conditional_flux = flux / continuum
                conditional_ivar = continuum * flux * continuum
                stacked_flux = np.sum(conditional_flux * conditional_ivar, axis=0) / np.sum(conditional_ivar, axis=0) 

                chi_sq, em_args = self._expectation_maximization(
                    flux, 
                    ivar, 
                    stacked_flux, 
                    phi, #phi, #None, # phi
                    continuum_args,
                )
                if iter > 0:
                    assert phi is not None
                if iter > 0 and (chi_sq > chi_sqs[-1]):                    
                    log.warning(f"Failed to improve \chi^2")
                    success, message = (False, "Failed to improve \chi^2")
                    break

                (phi, theta, continuum, model_rectified_flux, n_pixels, n_finite, n_nmf_iter) = em_args

                chi_sqs.append(chi_sq)
                n_pixels_used_in_nmf.append(n_pixels)
                n_pixels_used_in_chisq.append(n_finite)
                n_nmf_iters.append(n_nmf_iter)

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
            continuum_args=continuum_args,
            n_nmf_iters=n_nmf_iters
        )
        return (phi, theta, continuum, model_rectified_flux, meta)
