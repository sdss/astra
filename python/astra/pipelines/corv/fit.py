#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 09:06:32 2021

@author: vedantchandra
"""

import lmfit
import matplotlib.pyplot as plt
import numpy as np

from astra.pipelines.corv import utils
from astra.pipelines.corv import models

def normalized_residual(wl, fl, ivar, corvmodel, params, fit_window = None):
    """
    Error-scaled residuals between data and evaluated model

    Parameters
    ----------
    wl : array_like
        wavelengths in Angstroms.
    fl : array_like
        flux array.
    ivar : array_like
        inverse-variance.
    corvmodel : LMFIT Model class
        LMFIT model with normalization instructions.
    params : LMFIT Parameters class
        parameters at which to evaluate corvmodel.

    Returns
    -------
    resid : array_like
        error-scaled residual array.

    """
    
    nwl, nfl, nivar = utils.cont_norm_lines(wl, fl, ivar,
                                            corvmodel.names,
                                            corvmodel.centres,
                                            corvmodel.windows,
                                            corvmodel.edges)
        
    #if fit_window is not None:
    #    for ii in range(len(nivar)):
    #        in_center = []
    #                    
    #        for key in corvmodel.centres:
    #            center = corvmodel.centres[key]
    #            
    #            if (center - fit_window < wl[ii] < center + fit_window):
    #                in_center.append(True)
    #            else:
    #                in_center.append(False)
    #                
    #        if not True in in_center:
    #            nivar[ii] = 0            
    
    _,nmodel = models.get_normalized_model(wl, corvmodel, params)
    resid = (nfl - nmodel) * np.sqrt(nivar)
    
    return resid

def xcorr_rv(wl, fl, ivar, corvmodel, params,
             min_rv = -1500, max_rv = 1500, 
             npoints = 500,
             quad_window = 300, plot = False):
    """
    Find best RV via x-correlation on grid and quadratic fitting the peak.

    Parameters
    ----------
    wl : array_like
        wavelengths in Angstroms.
    fl : array_like
        flux array.
    ivar : array_like
        inverse-variance.
    corvmodel : LMFIT Model class
        LMFIT model with normalization instructions.
    params : LMFIT Parameters class
        parameters at which to evaluate corvmodel.
    min_rv : float, optional
        lower end of RV grid. The default is -1500.
    max_rv : float, optional
        upper end of RV grid. The default is 1500.
    resolution : float, optional
        resolution of the rv search grid in km/s. Default is 0.5 km/s.
    quad_window : float, optional
        window around minimum to fit quadratic model, 
        in km/s. The default is 300.

    Returns
    -------
    rv : float
        best-fit radial velocity.
    rvgrid : array_like
        grid of radial velocities.
    cc : array_like
        chi-square statistic evaluated at each RV.

    """
        
    rvgrid = np.linspace(min_rv, max_rv, npoints)
    cc = np.zeros(len(rvgrid))
    rcc = np.zeros(len(rvgrid))
    #params = corvmodel.make_params()
    
    residual = lambda params: normalized_residual(wl, fl, ivar, 
                                                  corvmodel, params, fit_window = 25)
    #print(params)
    for ii,rv in enumerate(rvgrid):
        params['RV'].set(value = rv)
        resid = residual(params)
        chi = np.nansum(resid**2)
        redchi = np.nansum(resid**2) / (len(resid) - 1)
        cc[ii] = chi
        rcc[ii] = redchi
        
    window = int(quad_window / np.diff(rvgrid)[0])

    # plt.plot(rvgrid, cc)
    # plt.show()
    
    argmin = np.nanargmin(cc)
    c1 = argmin - window

    if c1 < 0:
        c1 = 0

    c2 = argmin + window + 1

    #print(c1, c2)

    rvgrid = rvgrid[c1:c2]
    cc = cc[c1:c2]
    rcc = rcc[c1:c2]

    #plt.plot(rvgrid,rcc)
    
    try:
        pcoef = np.polyfit(rvgrid, cc, 2)
        rv = - 0.5 * pcoef[1] / pcoef[0]  
        
        t_cc = pcoef[0] * rv**2 + pcoef[1] * rv + pcoef[2]
        
        intersect = ( (-pcoef[1] + np.sqrt(pcoef[1]**2 - 4 * pcoef[0] * (pcoef[2] - t_cc - 1))) / (2 * pcoef[0]), 
                     (-pcoef[1] - np.sqrt(pcoef[1]**2 - 4 * pcoef[0] * (pcoef[2] - t_cc - 1))) / (2 * pcoef[0]) )
        
        e_rv = np.abs(intersect[0] - intersect[1]) / 2
        redchi = np.interp(rv, rvgrid, rcc)
    
        if plot:
            xgrid = np.linspace(min(rvgrid), max(rvgrid), 50)
            
            f = plt.figure(figsize = (10,5))
            pcoef = np.polyfit(rvgrid, cc, 2)
            plt.plot(rvgrid, cc, label = r'Actual $\chi^2$ curve')
            plt.plot(xgrid, pcoef[0]*xgrid**2 + pcoef[1]*xgrid + pcoef[2], label = r'Fitted $\chi^2$ curve')
            
            plt.axvline(x = rv)
            plt.axvline(x = rv + e_rv, ls = ':')
            plt.axvline(x = rv - e_rv, ls = ':')
            plt.axhline(y = t_cc, label = 'Minimum $\chi^2$')
            plt.legend()
            
        return rv, e_rv, redchi, rvgrid, cc
    except:
        print('pcoef failed!! returning min of chi function & err = 999')
        rv = rvgrid[np.nanargmin(cc)]
        e_rv = 999
        
    
    #print(t_cc)
    #print(cc)
    #print(temp)
    #
    #e_rv = (np.abs(min(rvgrid[temp]) -max(rvgrid[temp])) / 2)
    
        
    return rv, e_rv, redchi, rvgrid, cc

def fit_rv(wl, fl, ivar, corvmodel, params, fix_nonrv = True, 
           xcorr_kw = {}):
    """
    Use LMFIT to fit RV, after first estimating it by cross-correlation. 

    Parameters
    ----------
    wl : array_like
        wavelengths in Angstroms.
    fl : array_like
        flux array.
    ivar : array_like
        inverse-variance.
    corvmodel : LMFIT Model class
        LMFIT model with normalization instructions.
    params : LMFIT Parameters class
        parameters at which to evaluate corvmodel.
    fix_nonrv : book, optional
        whether to fix all non-RV parameters. The default is True.
    xcorr_kw : dict, optional
        keywords to pass to xcorr_rv. The default is {}.

    Returns
    -------
    res : LMFIT MinimzerResult class
        rv-fitting results.
    rv_init : float
        initial guess RV from the x-correlation, for comparison purposes.

    """
    
    rv, e_rv, redchi, rvgrid, cc = xcorr_rv(wl, fl, ivar, corvmodel, params,
                                   **xcorr_kw)
    
    #if fix_nonrv:
    #    for param in params:
    #        params[param].set(vary = False)
        
    #params['RV'].set(value = rv_init, vary = True)

    #residual = lambda params: normalized_residual(wl, fl, ivar, 
    #                                              corvmodel, params)
    
    #res = lmfit.minimize(residual, params)
    
    return rv, e_rv, redchi

def fit_corv(wl, fl, ivar, corvmodel, xcorr_kw = {},
                  iter_teff = False,
                  tpar = dict(tmin = 10000, tmax = 20000, nt = 2)):
    """
    Fit model parameters, x-corr RV, then LMFIT RV. 

    Parameters
    ----------
    wl : array_like
        wavelengths in Angstroms.
    fl : array_like
        flux array.
    ivar : array_like
        inverse-variance.
    corvmodel : LMFIT Model class
        LMFIT model with normalization instructions.
    xcorr_kw : dict, optional
        keywords to pass to xcorr_rv. The default is {}.
    iter_teff : bool, optional
        whether to iterate over several initial teffs. The default is False.
    tpar : dict, optional
        initial teff iteration parameters. The default is 
        dict(tmin = 10000, tmax = 20000, nt = 2).

    Returns
    -------
    param_res : LMFIT MinimizerResult class
        result of fit to parameters, allowing everything to vary.
    rv_res : LMFIT MinimizerResult class
        result of RV fit from LMFIT.
    rv_init : float
        best RV from x-correlation, in km/s

    """
    
    params = corvmodel.make_params()
    
    residual = lambda params: normalized_residual(wl, fl, ivar, 
                                                  corvmodel, params)
    
    if iter_teff:
        minchi = 1e50
        init_teffs = np.linspace(tpar['tmin'], tpar['tmax'], tpar['nt'])
        for ii in range(tpar['nt']):
            params_i = params.copy()
            params_i['teff'].set(value = init_teffs[ii])
            resi = lmfit.minimize(residual, params_i)
            
            if resi.redchi < minchi:
                param_res = resi
                minchi = resi.redchi
            else:
                continue
    else:
        param_res = lmfit.minimize(residual, params)
        
    bestparams = param_res.params.copy()
    
    rv, e_rv, redchi = fit_rv(wl, fl, ivar, corvmodel, bestparams, **xcorr_kw)
            
    return rv, e_rv, redchi, param_res
