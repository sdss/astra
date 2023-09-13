#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 12:08:00 2021

@author: vedantchandra
"""

import numpy as np
from bisect import bisect_left
import scipy
import matplotlib.pyplot as plt
from astropy import constants as c

def lineplot(wl, fl, ivar, corvmodel, params, gap = 0.3, printparams = True,
             figsize = (6, 5)):
    
    model = corvmodel.eval(params, x = wl)
    
    chi2 = 0
    dof = 0
    
    f = plt.figure(figsize = figsize)
    
    for ii,line in enumerate(corvmodel.names):
        
        cwl, cfl, civar = cont_norm_line(wl, fl, ivar, 
                                     corvmodel.centres[line],
                                     corvmodel.windows[line],
                                     corvmodel.edges[line])
        if len(cwl) == 0:
            continue

        _, cmodel, _ = cont_norm_line(wl, model, model, 
                                     corvmodel.centres[line],
                                     corvmodel.windows[line],
                                     corvmodel.edges[line])
        
        dlam = (cwl - corvmodel.centres[line])
        
        plt.plot(dlam, cfl - ii * gap, 'k')
        plt.plot(dlam, cmodel - ii * gap, 'r')
        
        chi2 += np.sum((cfl - cmodel)**2 * civar)
        dof += len(cfl)
        
    redchi = chi2 / (dof - len(params))
        
    plt.xlabel(r'$\mathrm{\Delta \lambda}\ (\mathrm{\AA})$')
    plt.ylabel('Normalized Flux')
    
    if printparams:
    
        plt.text(0.97, 0.05, 
                 r'$T_{\mathrm{eff}} = %.0f \pm %.0f\ K$' % 
                 (params['teff'].value, params['teff'].stderr or np.nan),
    			transform = plt.gca().transAxes, fontsize = 14, ha = 'right')
    		
        plt.text(0.97, 0.12, 
                 r'$\log{g} = %.2f \pm %.2f $' % 
                 (params['logg'].value, params['logg'].stderr or np.nan),
    			transform = plt.gca().transAxes, fontsize = 14, ha = 'right')
    				 
        plt.text(0.97, 0.19, r'$\chi_r^2$ = %.2f' % (redchi),
    			transform = plt.gca().transAxes, fontsize = 14, ha = 'right')
        
        
    plt.ylim(-gap * ii, 1 + gap)
    
    return f
    


def cont_norm_line(wl, fl, ivar, centre, window, edge):
    """
    Continuum-normalizes a single absorption/emission line.

    Parameters
    ----------
    wl : array_like
        wavelength.
    fl : array_like
        flux.
    ivar : array_like
        inverse-variance.
    centre : float
        line centroid.
    window : int
        selected region on either side of line, in pixels.
    edge : int
        number of pixels on edge of region used to define continuum.

    Returns
    -------
    wl : array_like
        cropped wavelength array.
    norm_fl : array_like
        cropped and normalized flux array.
    norm_ivar : array_like
        cropped and normalized inverse-variance array.

    """
    c1 = bisect_left(wl, centre - window)
    c2 = bisect_left(wl, centre + window)
    wl, fl, ivar = wl[c1:c2], fl[c1:c2], ivar[c1:c2]

    mask = np.ones(len(wl))
    mask[edge:-edge] = 0
    mask = mask.astype(bool)
    if not any(mask):
        return ([], [], [])

    p = np.polynomial.polynomial.polyfit(wl[mask], fl[mask], 1)
    continuum = np.polynomial.polynomial.polyval(wl, p)
    norm_fl = fl / continuum
    norm_ivar = ivar * continuum**2
    return wl, norm_fl, norm_ivar

def cont_norm_lines(wl, fl, ivar, names, centres, windows, edges):
    nwl = []
    nfl = []
    nivar = []
    
    for line in names:
        nwli, nfli, nivari = cont_norm_line(wl, 
                                            fl, 
                                            ivar, 
                                            centres[line], 
                                            windows[line], 
                                            edges[line])
        nwl.extend(nwli)
        nfl.extend(nfli)
        nivar.extend(nivari)
        
    return np.array(nwl), np.array(nfl), np.array(nivar)



def crrej(wl, fl, ivar, nsig = 3, medwindow = 11, plot = False):

    medfl = scipy.ndimage.median_filter(fl, medwindow)
    
    if plot:
        plt.plot(wl, fl)
        plt.plot(wl, medfl)
        plt.show()
    
    zscore = (fl  - medfl) * np.sqrt(ivar)

    crmask = (np.abs(zscore) > nsig) | (ivar == 0)
    
    corr_ivar = ivar
    corr_ivar[crmask] = 0
    corr_fl = np.interp(wl, wl[~crmask], fl[~crmask])
    
    if plot:
        plt.title('crrej z-score')
        plt.plot(wl, np.abs(zscore))
        plt.ylim(-0.5, 5)
        plt.show()
        print('%i pixels rejected' % np.sum(crmask))

    return wl, corr_fl, corr_ivar

def air2vac(wv):
    """
    Air to vacuum wavelengths, formula from Morton 1991 ApJS, 77, 119.

    Parameters
    ----------
    wv : array_like
        air wavelengths in Angstrom.

    Returns
    -------
    arary_like
        vacuum wavelengths in Angstrom. 

    """
    _tl=1.e4/np.array(wv)
    return (np.array(wv)*(1.+6.4328e-5+2.94981e-2/\
                          (146.-_tl**2)+2.5540e-4/(41.-_tl**2)))

def vac2air(wv):
    """
    Vacuum to air wavelengths, formula from Morton 1991 ApJS, 77, 119.

    Parameters
    ----------
    wv : array_like
        vacuum wavelengths in Angstrom.

    Returns
    -------
    arary_like
        air wavelengths in Angstrom. 

    """
    _tl = 1.e4/np.array(wv)
    return (np.array(wv) / (1. + 6.4328e-5 + 2.94981e-2
                            / (146. - _tl**2) + 2.5540e-4 / (41. - _tl**2)))

def doppler_shift(wl, fl, dv):
       c = 2.99792458e5
       df = np.sqrt((1 - dv/c)/(1 + dv/c)) 
       new_wl = wl * df
       new_fl = np.interp(new_wl, wl, fl)
       return new_fl
        
def get_medsn(wl, fl, ivar):
    wlsel = (wl > 5400) & (wl < 5800)
    cwl, cfl, civar = wl[wlsel], fl[wlsel], ivar[wlsel]
    medsn = np.nanmedian(cfl * np.sqrt(civar))
    contnorm = cfl / np.polyval(np.polyfit(cwl, cfl, 2), cwl)
    sigma_est = np.std(contnorm)
    
    return medsn, 1/sigma_est
