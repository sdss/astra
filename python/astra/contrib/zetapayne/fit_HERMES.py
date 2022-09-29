import sys,os
from math import *
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy import interpolate
from bisect import bisect
from DER_SNR import DER_SNR
from common import param_names, param_units
from Network import Network
from Fit import Fit
from numpy.polynomial.chebyshev import chebval
from fit_common import save_figure
from UncertFit import UncertFit
from multiprocessing import Pool, Lock


lock = Lock()

def load_spectrum(fn):

   # Read the data and the header is resp. 'spec' and 'header'
   flux = fits.getdata(fn)
   header = fits.getheader(fn)
   #
   # Make the equidistant wavelengthgrid using the Fits standard info
   # in the header
   #
   ref_pix = int(header['CRPIX1'])-1
   ref_val = float(header['CRVAL1'])
   ref_del = float(header['CDELT1'])
   numberpoints = flux.shape[0]
   unitx = header['CTYPE1']
   wavelengthbegin = ref_val - ref_pix*ref_del
   wavelengthend = wavelengthbegin + (numberpoints-1)*ref_del
   wavelengths = np.linspace(wavelengthbegin,wavelengthend,numberpoints)
   wavelengths = np.exp(wavelengths)

   return wavelengths, flux

def multiplot(wave, flux, N, title, lbl, xlbl, ylbl):
    di = len(wave)//N
    for i in range(N):
        i1 = i*di
        i2 = (i+1)*di
        plt.subplot(100*N + i + 11)
        if i==0: plt.title(title)
        plt.plot(wave[i1:i2], flux[i1:i2], label=lbl)
        if ylbl!=None: plt.ylabel(ylbl)
        if i==N-1:
            if xlbl!=None: plt.xlabel(xlbl)
            plt.legend()

def plot_slice(fit_res, NN, N, par1, par2):
    xx = np.linspace(-0.5, 0.5, N)
    zz = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            popt = fit_res.popt_scaled
            popt[par1] = xx[i]
            popt[par2] = xx[j]
            zz[i,j] = fit_res.chi2_func(popt)

    grid = NN.grid

    pn = param_names
    name1 = pn[par1]
    name2 = pn[par2]
    range1 = grid[name1]
    range2 = grid[name2]
    ext = (range1[0], range1[1], range2[0], range2[1])

    im = plt.imshow(zz, origin='lower', interpolation='none', aspect='auto', extent=ext)
    plt.colorbar(im)
    plt.xlabel(name1)
    plt.ylabel(name2)
    fn = name1 + '-' + name2 + '.png'
    fn = fn.replace('/', '|')
    plt.savefig('FIT/SLICE_'+fn)
    plt.clf()

def plot_slices(fit_res, NN, N):
    num_lab = NN.num_labels()
    for i in range(num_lab):
        for j in range(i+1, num_lab):
            plot_slice(fit_res, NN, N, i, j)

def get_indices(wave, w1, w2):
   i1 = bisect(wave, w1)
   i2 = bisect(wave, w2)
   return i1, i2

def get_path(night, seq_id):
    return '/STER/mercator/hermes/'+night+'/reduced/'+seq_id.zfill(8)+'_HRF_OBJ_ext_CosmicsRemoved_log_merged_cf.fits'

def fit_HERMES(night, seq_id, NN, wave_start, wave_end, Cheb_order=5, constraints={}):
    fn = get_path(night, seq_id)
    wave, flux = load_spectrum(fn)
    hdr = fits.getheader(fn)
    obj_name = hdr['OBJECT']

    start_idx = bisect(wave, wave_start)
    end_idx = bisect(wave, wave_end)
    wave = wave[start_idx:end_idx]
    flux = flux[start_idx:end_idx]
    flux_original = np.copy(flux)
    SNR = DER_SNR(flux)
    flux_mean = np.mean(flux)
    flux /= flux_mean
    err = flux / SNR

    grid_params = [p for p in param_names if NN.grid[p][0]!=NN.grid[p][1]]
    bounds_unscaled = np.zeros((2, len(grid_params)))
    for i,v in enumerate(grid_params):
        if v in constraints:
            bounds_unscaled[0,i] = constraints[v][0]
            bounds_unscaled[1,i] = constraints[v][1]
        else:
            bounds_unscaled[0,i] = NN.grid[v][0]
            bounds_unscaled[1,i] = NN.grid[v][1]


    fit = Fit(NN, Cheb_order)
    fit.bounds_unscaled = bounds_unscaled
    fit.N_presearch_iter = 1
    fit.N_pre_search = 4000
    unc_fit = UncertFit(fit, 85000)
    fit_res = unc_fit.run(wave, flux, err)


    fit_res.wave = wave
    fit_res.flux = flux_original
    fit_res.model *= flux_mean
    fit_res.obj_name = obj_name
    return fit_res


    

if __name__=='__main__':

    if len(sys.argv)<4:
        print('Use:', sys.argv[0], '<night> <sequence_id> <start_wavelength> <end_wavelength>')
        exit()

    night = sys.argv[1]
    seq_id = sys.argv[2]
    wave_start = float(sys.argv[3])
    wave_end = float(sys.argv[4])
    slices = 'slices' in sys.argv

    NN_path = '/STER/ilyas/inbox/NN_n1000_b512_v0.1_OPTIC_T5k_10k.npz'
    NN = Network()
    NN.read_in(NN_path)

    fit_HERMES(night, seq_id, NN, wave_start, wave_end, Cheb_order=5, slices=slices)
    








