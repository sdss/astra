import sys,os
import numpy as np
import matplotlib.pyplot as plt
#import emcee # TODO
#import corner # TODO
from multiprocessing import Pool
#from Fit import Fit
from astra.contrib.zetapayne.Fit import Fit

def lnlike(x, data):
    chi2 = data[0]
    return -0.5 * chi2(x)

def lnprior(x):
    return 0.0

def lnprob(x, data):
    lp = lnprior(x)
    if not np.isfinite(lp):
        return -np.inf
    return np.array([lp + lnlike(x, data)])


class MCMC_Fit:

    def __init__(self, fit:Fit):
        self.fit = fit
        self.nwalkers = 100
        self.nsamples = 1000
        self.burn_in = 100
        
    def run(self, wave, flux, flux_err):
        che = self.fit.Cheb_order
    
        popt, pcov, model_spec, chi2_func = self.fit.run(wave, flux, flux_err)
        opt = popt
        
        ndim = len(opt)
        nwalkers, nsampl, burn_in = self.nwalkers, self.nsamples, self.burn_in
        pos = [ opt + 1.e-5*np.random.rand(ndim) for i in range(nwalkers)]

        #pool = Pool(processes=4)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=([chi2_func, popt],))#, pool=pool)
        #os.environ['OMP_NUM_THREADS'] = '1'
        # TODO: parallel sampling, requires serializable data
        print('Start sampling')
        sampler.run_mcmc(pos, nsampl, progress=True)

        print('Acceptance fractions:')
        print(sampler.acceptance_fraction)
        #print('Autocorr. time:', np.mean(sampler.get_autocorr_time()))
        
        samples = sampler.chain[:, burn_in:, :].reshape((-1, ndim))
        MAP_est = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0))))

        fig = corner.corner(samples)
        fig.savefig('FIT/triangle.png')
        plt.clf()
        
        return popt, MAP_est, model_spec, chi2_func


