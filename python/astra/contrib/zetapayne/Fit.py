# code for fitting spectra, using the models in spectral_model.py
from __future__ import absolute_import, division, print_function # python2 compatibility
import math
import numpy as np
from scipy.optimize import curve_fit
from bisect import bisect
from numpy.polynomial.chebyshev import chebval
#from Network import Network
from scipy.ndimage import gaussian_filter1d
#import sobol_seq
from astra.contrib.zetapayne.Network import Network
from astra.contrib.zetapayne import sobol_seq

def doppler_shift(wavelength, flux, dv):
    '''
    dv is in km/s
    We use the convention where a positive dv means the object is moving away.

    This linear interpolation is actually not that accurate, but is fine if you
    only care about accuracy to the level of a few tenths of a km/s. If you care
    about better accuracy, you can do better with spline interpolation.
    '''
    c = 2.99792458e5 # km/s
    doppler_factor = np.sqrt((1 - dv/c)/(1 + dv/c))
    new_wavelength = wavelength * doppler_factor
    new_flux = np.interp(new_wavelength, wavelength, flux)
    return new_flux

def gaussian(xx, mu, sigma):
    return np.exp(-0.5*((xx-mu)/sigma)**2)/(sigma * math.sqrt(2*math.pi))

class FitResult:
    pass

class GaussKernel:

    def __init__(self, wave, w0, sigma):
        Q = 5.0
        w_start = w0 - Q*sigma
        w_end = w0 + Q*sigma
        i_start = bisect(wave, w_start)
        i_end = bisect(wave, w_end)

        wave_slice = wave[i_start:i_end]
        self.i_start = i_start
        self.i_end = i_end
        self.wave = wave_slice
        self.kernel = gaussian(wave_slice, w0, sigma)

    def integrate(self, flux):
        return np.trapz(self.kernel * flux[self.i_start:self.i_end], self.wave)


class Fit:

    def __init__(self, network:Network, Cheb_order, tol=5.e-4):
        self.tol = tol # tolerance for when the optimizer should stop optimizing.
        self.network = network
        self.Cheb_order = Cheb_order
        self.N_presearch_iter = 0
        self.N_pre_search = 2000
        self.RV_presearch_range = 100.0 # Search RV in [-50,50] km/s

    def run(self, wavelength, norm_spec, spec_err, mask=None, p0 = None):
        '''
        fit a single-star model to a single combined spectrum

        p0 is an initial guess for where to initialize the optimizer. Because
            this is a simple model, having a good initial guess is usually not
            important.

        labels = [Teff, Logg, Vturb [km/s],
                [C/H], [N/H], [O/H], [Na/H], [Mg/H],\
                [Al/H], [Si/H], [P/H], [S/H], [K/H],\
                [Ca/H], [Ti/H], [V/H], [Cr/H], [Mn/H],\
                [Fe/H], [Co/H], [Ni/H], [Cu/H], [Ge/H],\
                C12/C13, Vmacro [km/s], radial velocity

        returns:
            popt: the best-fit labels
            pcov: the covariance matrix, from which you can get formal fitting uncertainties
            model_spec: the model spectrum corresponding to popt
        '''

        # set infinity uncertainty to pixels that we want to omit
        if mask != None:
            spec_err[mask] = 999.

        # number of labels + radial velocity
        nnl = self.network.num_labels()
        num_labels = nnl + self.Cheb_order + 1


        def fit_func(dummy_variable, *labels):
            nn_spec = self.network.get_spectrum_scaled(scaled_labels = labels[:nnl])
            nn_spec = doppler_shift(self.network.wave, nn_spec, labels[nnl])
            if hasattr(self, 'lsf'):
                nn_resampl = self.lsf.apply(nn_spec)
            else:
                nn_resampl = np.interp(wavelength, self.network.wave, nn_spec)
            Cheb_coefs = labels[nnl + 1 : nnl + 1 + self.Cheb_order]
            Cheb_x = np.linspace(-1, 1, len(nn_resampl))
            Cheb_poly = chebval(Cheb_x, Cheb_coefs)
            spec_with_resp = nn_resampl * Cheb_poly
            return spec_with_resp

        # if no initial guess is supplied, initialize with the median value
        if p0 is None:
            p0 = np.zeros(num_labels)

        x_min = self.network.x_min
        x_max = self.network.x_max

        # prohibit the minimimizer to go outside the range of training set
        bounds = np.zeros((2,num_labels))
        if not hasattr(self, 'bounds_unscaled'):
            bounds[0,:nnl] = -0.5
            bounds[1,:nnl] = 0.5
        else:
            for i in [0,1]:
                bounds[i,:nnl] = (self.bounds_unscaled[i,:]-x_min)/(x_max-x_min) - 0.5
        bounds[0, nnl:] = -np.inf
        bounds[1, nnl:] = np.inf

        # make sure the starting point is within bounds
        for i in range(num_labels):
            if not bounds[0,i]  < p0[i] < bounds[1,i]:
                p0[i] = 0.5*(bounds[0,i] + bounds[1,i])

        # pre-searching on sobol grid
        for it in range(self.N_presearch_iter):
            popt, pcov = curve_fit(fit_func, xdata=[], ydata = norm_spec, sigma = spec_err, p0 = p0,
                        bounds = bounds, ftol = self.tol, xtol = self.tol, absolute_sigma = True, method = 'trf')

            best = [np.inf, None]
            N_sobol = sobol_seq.i4_sobol_generate(nnl+1, self.N_pre_search)
            for i in range(self.N_pre_search):
                lbl = np.copy(popt)
                lbl[:nnl] = bounds[0,:nnl] + N_sobol[i,:nnl]*(bounds[1,:nnl] - bounds[0,:nnl])
                lbl[nnl] = self.RV_presearch_range * (N_sobol[i,nnl] - 0.5) #RV

                model = fit_func([], *lbl)
                diff = (norm_spec - model) / spec_err
                chi2 = np.sum(diff**2)
                if chi2 < best[0]:
                    best[0] = chi2
                    best[1] = lbl

            p0 = best[1]

        # run the optimizer
        popt, pcov = curve_fit(fit_func, xdata=[], ydata = norm_spec, sigma = spec_err, p0 = p0,
                    bounds = bounds, ftol = self.tol, xtol = self.tol, absolute_sigma = True, method = 'trf')
        model_spec = fit_func([], *popt)

        res = FitResult()
        res.model = model_spec
        res.popt_scaled = np.copy(popt)
        res.pcov_scaled = np.copy(pcov)

        # rescale the result back to original unit
        popt[:nnl] = (popt[:nnl]+0.5)*(x_max-x_min) + x_min
        pcov[:nnl,:nnl] = pcov[:nnl,:nnl]*(x_max-x_min)

        res.popt = popt
        res.pcov = pcov
        
        def chi2_func(labels):
            labels_sc = np.copy(labels)
            labels_sc[:nnl] = (labels_sc[:nnl] - x_min)/(x_max - x_min) - 0.5
            model = fit_func([], *labels_sc)
            diff = (norm_spec - model) / spec_err
            chi2 = np.sum(diff**2)
            return chi2
        
        res.chi2_func = chi2_func
        return res












