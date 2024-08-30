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

import re
import pickle
from scipy.interpolate import RegularGridInterpolator
from astropy.table import Table
import glob
import os

from . import models
from tqdm import tqdm

#plt.style.use('./stefan.mplstyle')

def lineplot(wl, fl, ivar, corvmodel, params, gap = 0.3, printparams = True,
             figsize = (10, 7)):
    
    model = corvmodel.eval(params, x = wl)
    
    chi2 = 0
    dof = 0
    
    f = plt.figure(figsize = figsize)
    
    
    for ii,line in enumerate(corvmodel.names):
        
        cwl, cfl, civar = cont_norm_line(wl, fl, ivar, 
                                     corvmodel.centres[line],
                                     corvmodel.windows[line],
                                     corvmodel.edges[line])
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

    p = np.polynomial.polynomial.polyfit(wl[mask], fl[mask], 1)
    continuum = np.polynomial.polynomial.polyval(wl, p)
    norm_fl = fl / continuum
    norm_ivar = ivar * continuum**2
    return wl, norm_fl, norm_ivar

def cont_norm_lines(wl, fl, ivar, names, centres, windows, edges):
    nwl = [];
    nfl = [];
    nivar = [];
    
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

basepath = os.path.dirname(os.path.abspath(__file__))

def build_montreal_da(path, outpath = None, flux_unit = 'fnu'):
    files = glob.glob(path + '/*')
    with open(files[0]) as f:
        lines = f.read().splitlines()
    
    table = Table()
    dat = []
    
    with open(files[0]) as f:
        lines = f.read().splitlines()
        
    base_wavl = []
        
    for ii in range(len(lines)):        
        if 'Effective temperature' in lines[ii]:
            base_wavl = np.array(re.split('\s+', ''.join(lines[1:ii])))[1:].astype(float)
            break                
                
    for file in files:
        with open(file) as f:
            lines = f.read().splitlines()
                
        prev_ii = 0
            
        first = True
        
        for ii in range(len(lines)):        
            if 'Effective temperature' in lines[ii]:
                if first:
                    wavl = np.array(re.split('\s+', ''.join(lines[1:ii])))[1:].astype(float)
                    first = False
                    prev_ii = ii
                    continue
                    
                #print(prev_ii)
                                
                teff = float(re.split('\s+', lines[prev_ii])[4])
                logg = np.log10(float(re.split('\s+', lines[prev_ii])[7]))
                
                if not first:
                    fl = re.split('\s+', ''.join(lines[prev_ii+1:ii]))
                    for jj, num in enumerate(fl):
                        if 'E' not in num:
                            if '+' in num:
                                num = num.split('-')
                                num = 'E'.join(num)
                            elif ('-' in num) and (num[0] != '-'):
                                num = num.split('-')
                                num = 'E-'.join(num)
                            elif ('-' in num) and (num[0] == '-'):
                                num = num.split('-')
                                num = 'E-'.join(num)
                                num = num[1:]
                            fl[jj] = num
                    try:
                        fl = np.array([float(val) for val in fl])
                    except:
                        fl = fl[1:]
                        fl = np.array([float(val) for val in fl])
                        
                    dat.append([logg, teff, np.interp(base_wavl, wavl, fl), base_wavl])                                                   
                    #fls[str(teff) + ' ' + str(logg)] = fl
                    
                    prev_ii = ii
        
        
    default_centres =  dict(a = 6564.61, b = 4862.68, g = 4341.68, d = 4102.89,
                     e = 3971.20, z = 3890.12, n = 3835.5,
                 t = 3799.5)
    default_windows = dict(a = 100, b = 100, g = 85, d = 70, e = 30,
                      z = 25, n = 15, t = 10)
    default_edges = dict(a = 25, b = 25, g = 20, d = 20, 
                    e = 5, z = 5, n = 5, t = 4)
    
    default_names = ['n', 'z', 'e', 'd', 'g', 'b', 'a']
                
    table['teff'] = np.array(dat, dtype=object).T[1]
    table['logg'] = np.array(dat, dtype=object).T[0]
    wavls = air2vac(np.array(dat, dtype=object).T[3])
    fls = np.array(dat, dtype=object).T[2] # convert from erg cm^2 s^1 Hz^-1 ---> erg cm^2 s^1 A^-1
    #ivar = 1e10*np.zeros((len(fls), len(fls[0])))
        
    #test = np.array([cont_norm_lines(wavls[i], fls[i], ivar[i], default_names, default_centres, default_windows, default_edges) for i in range(len(wavls))])
    
    #wavl = np.linspace(3600, 9000, 3747)
    #wavl_arr = [wavl for i in range(len(table))]
    
    #fl_arr = []
    #for i in tqdm(range(len(table))):
    #    fl_arr.append(np.interp(wavl, test[i][0], test[i][1]))
    
    table['wl'] = wavls
    table['fl'] = fls
    if flux_unit == 'flam':
        table['fl'] = (2.99792458e18*table['fl'] / table['wl']**2)

    
    #table['fl'] = [cont_norm_lines(table['wl'][i], table['fl'][i], avg_size = 400)[1] for i in tqdm(range(len(table)))]
    #table['fl'] = [_cont_norm(table['fl'], np.zeros((len(table['fl']), len(table['fl'][0]))), np.ones((len(table['fl']), len(table['fl'][0]))) )[1] for i in tqdm(range(len(table)))]
    
    teffs = sorted(list(set(table['teff'])))
    loggs = sorted(list(set(table['logg'])))
    
    values = np.zeros((len(teffs), len(loggs), 3747))
    
    for i in range(len(teffs)):
        for j in range(len(loggs)):
            try:
                values[i,j] = table[np.all([table['teff'] == teffs[i], table['logg'] == loggs[j]], axis = 0)]['fl'][0]
            except:
                values[i,j] = np.zeros(3747)
    
    #NICOLE BUG FIX
    high_logg_grid=values[:,4:]
    high_loggs=loggs[4:]

    low_logg_grid=values[16:33,:]
    low_loggs_teffs=teffs[16:33]

    model_spec = RegularGridInterpolator((teffs, high_loggs), high_logg_grid)
    model_spec_low_logg = RegularGridInterpolator((low_loggs_teffs, loggs), low_logg_grid)
    
    if outpath is not None:
        # open a file, where you ant to store the data
        interp_file = open(outpath + '/montreal_da.pkl', 'wb')
        
        # dump information to that file
        pickle.dump(model_spec, interp_file)
        np.save(outpath + '/montreal_da_wavl', base_wavl)
        
    return wavls[0], model_spec, model_spec_low_logg, table


def build_montreal_db(path, outpath = None):
    table = Table()
    dat = []
    files = glob.glob(path + '/*')
    
    with open(files[0]) as f:
        lines = f.read().splitlines()
        
    base_wavl = []
        
    for ii in range(len(lines)):        
        if 'Effective temperature' in lines[ii]:
            base_wavl = np.array(re.split('\s+', ''.join(lines[1:ii])))[1:].astype(float)
            break   
    
    for file in files:
        with open(file) as f:
            lines = f.read().splitlines()
                
        prev_ii = 0
            
        first = True
            
        for ii in range(len(lines)):        
            if 'Effective temperature' in lines[ii]:
                if first:
                    wavl = np.array(re.split('\s+', ''.join(lines[1:ii])))[1:].astype(float)
                    first = False
                    prev_ii = ii
                    continue
                                    
                teff = float(re.split('\s+', lines[prev_ii])[4])
                logg = np.log10(float(re.split('\s+', lines[prev_ii])[7]))
                he = float(re.split('\s+', lines[prev_ii])[-1])
                            
                if not first:
                    fl = re.split('\s+', ''.join(lines[prev_ii+1:ii]))
                    for jj, num in enumerate(fl):
                        if 'E' not in num:
                            if '+' in num:
                                num = num.split('-')
                                num = 'E'.join(num)
                            elif ('-' in num) and (num[0] != '-'):
                                num = num.split('-')
                                num = 'E-'.join(num)
                            elif ('-' in num) and (num[0] == '-'):
                                num = num.split('-')
                                num = 'E-'.join(num)
                                num = num[1:]
                            fl[jj] = num
                    try:
                        fl = np.array([float(val) for val in fl])
                    except:
                        fl = fl[1:]
                        fl = np.array([float(val) for val in fl])
                        
                    dat.append([logg, teff, he, np.interp(base_wavl, wavl, fl)])                                                   
                    #fls[str(teff) + ' ' + str(logg)] = fl
                    
                    prev_ii = ii
                    
    table['logg'] = np.array(dat).T[0]
    table['teff'] = np.array(dat).T[1]
    table['y'] = np.array(dat).T[2]
    table['fl'] = [continuum_normalize(base_wavl, np.array(dat).T[3][i]) for i in tqdm(range(len(np.array(dat).T[3])))]

    teffs = sorted(list(set(table['teff'])))
    loggs = sorted(list(set(table['logg'])))
    hes = sorted(list(set(table['y'])))
    
    values = np.zeros((len(teffs), len(loggs), len(hes), 2711))
    
    for i in range(len(teffs)):
        for j in range(len(loggs)):
            for k in range(len(hes)):
                try:
                    values[i,j,k] = table[np.all([table['teff'] == teffs[i], table['logg'] == loggs[j], table['y'] == hes[k]], axis = 0)]['fl'][0]
                except:
                    values[i,j,k] = np.zeros(2711)
    
    model_spec = RegularGridInterpolator((teffs, loggs, hes), values)
    
    if outpath is not None:
        # open a file, where you ant to store the data
        interp_file = open(outpath + '/montreal_db.pkl', 'wb')
        
        # dump information to that file
        pickle.dump(model_spec, interp_file)
        np.save(outpath + '/montreal_db_wavl', base_wavl)
    
    return base_wavl, model_spec

def continuum_normalize(wl, fl, ivar = None, avg_size = 150, ret_cont = False):
    
    fl_norm = np.zeros(np.size(fl))
    fl_cont = np.zeros(np.size(fl))
    
    ivar_yes = 0
    if ivar is not None:
        ivar_yes = 1
        ivar_norm = np.zeros(np.size(fl))
        
    for i in range(np.size(wl)):
        wl_clip = ((wl[i]-avg_size/2)<wl) * (wl<(wl[i]+avg_size/2))
        fl_cont[i] = np.median(fl[wl_clip])
        if ivar_yes:
            ivar_norm[i] = ivar[i]*np.median(fl[wl_clip])**2
    
    fl_norm = fl/fl_cont
    
    if ret_cont:
        if ivar_yes:
            return wl, fl_norm, ivar_norm, fl_cont
        else:
            return wl, fl_norm, fl_cont
    else:
        if ivar_yes:
            return wl, fl_norm, ivar_norm
        else:
            return wl, fl_norm
