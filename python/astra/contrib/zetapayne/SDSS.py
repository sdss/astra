#!/usr/bin/env python
#
# Utility for reading in the SDSS-V MWM training set spectra
#

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@noao.edu>'
__version__ = '20180912'  # yyyymmdd

import re
import logging
import os
import numpy as np
from astropy.io import fits
from astropy.table import Table, Column

def flatten(a):
    return np.flip(a.flatten())

# Object for representing 1D spectra
class Spec1D:
    # Initialize the object
    def __init__(self,flux):
        self.flux = flux
        return

    def __repr__(self):
        s = repr(self.__class__)+"\n"
        s += self.instrument+" "+self.sptype+" "+self.waveregime+" spectrum\n"
        s += "File = "+self.filename+"\n"
        s += ("S/N = %7.2f" % self.snr)+"\n"
        s += "Flux = "+str(self.flux)+"\n"
        s += "Err = "+str(self.err)+"\n"
        s += "Wave = "+str(self.wave)
        return s

def rdspec(filename=None):
    '''
    This reads in a SDSS-IV MWM training set spectrum and returns an
    object that is guaranteed to have certain information.

    Parameters
    ----------
    filename : str
          The filename of the spectrum.

    Returns
    -------
    spec : Spec1D object
        A Spec1D object that always has FLUX, ERR, WAVE, MASK, FILENAME, SPTYPE, WAVEREGIME and INSTRUMENT.

    Example
    -------

    Load an APOGEE apStar spectrum.

    .. code-block:: python

        spec = rdspec("apStar-r8-2M00050083+6349330.fits")

    '''
    if filename is None:
        print("Please input the filename")
        return

    # Different types of spectra
    # apVisit
    # apStar
    # BOSS spec
    # MaStar spectra
    # lp*fits, synthetic spectra
    base, ext = os.path.splitext(os.path.basename(filename))

    # Check that the files exists
    if os.path.exists(filename) is False:
        print(filename+" NOT FOUND")
        return None
    
    # APOGEE apVisit, visit-level spectrum
    if base.find("apVisit") > -1 or base.find("asVisit") > -1:
        flux = flatten(fits.getdata(filename,1))
        spec = Spec1D(flux)
        spec.filename = filename
        spec.sptype = "apVisit"
        spec.waveregime = "NIR"
        spec.instrument = "APOGEE"        
        spec.head = fits.getheader(filename,0)
        spec.err = flatten(fits.getdata(filename,2))
        spec.mask = flatten(fits.getdata(filename,3))
        spec.wave = flatten(fits.getdata(filename,4))
        spec.sky = flatten(fits.getdata(filename,5))
        spec.skyerr = flatten(fits.getdata(filename,6))
        spec.telluric = flatten(fits.getdata(filename,7))
        spec.telerr = flatten(fits.getdata(filename,8))
        spec.wcoef = fits.getdata(filename,9)
        spec.lsf = fits.getdata(filename,10)
        spec.meta = fits.getdata(filename,11)   # catalog of RV and other meta-data
        # Spectrum, error, sky, skyerr are in units of 1e-17
        spec.snr = spec.head["SNR"]
        return spec

    # APOGEE apStar, combined spectrum
    if base.find("apStar") > -1 or base.find("asStar") > -1:
        # HISTORY APSTAR:  HDU0 = Header only                                             
        # HISTORY APSTAR:  All image extensions have:                                     
        # HISTORY APSTAR:    row 1: combined spectrum with individual pixel weighting     
        # HISTORY APSTAR:    row 2: combined spectrum with global weighting               
        # HISTORY APSTAR:    row 3-nvisits+2: individual resampled visit spectra          
        # HISTORY APSTAR:   unless nvisits=1, which only have a single row                
        # HISTORY APSTAR:  All spectra shifted to rest (vacuum) wavelength scale          
        # HISTORY APSTAR:  HDU1 - Flux (10^-17 ergs/s/cm^2/Ang)                           
        # HISTORY APSTAR:  HDU2 - Error (10^-17 ergs/s/cm^2/Ang)                          
        # HISTORY APSTAR:  HDU3 - Flag mask:                                              
        # HISTORY APSTAR:    row 1: bitwise OR of all visits                              
        # HISTORY APSTAR:    row 2: bitwise AND of all visits                             
        # HISTORY APSTAR:    row 3-nvisits+2: individual visit masks                      
        # HISTORY APSTAR:  HDU4 - Sky (10^-17 ergs/s/cm^2/Ang)                            
        # HISTORY APSTAR:  HDU5 - Sky Error (10^-17 ergs/s/cm^2/Ang)                      
        # HISTORY APSTAR:  HDU6 - Telluric                                                
        # HISTORY APSTAR:  HDU7 - Telluric Error                                          
        # HISTORY APSTAR:  HDU8 - LSF coefficients                                        
        # HISTORY APSTAR:  HDU9 - RV and CCF structure  
        flux = fits.getdata(filename,1)
        spec = Spec1D(flux)
        spec.filename = filename
        spec.sptype = "apStar"
        spec.waveregime = "NIR"
        spec.instrument = "APOGEE"
        spec.head = fits.getheader(filename,0)
        spec.err = fits.getdata(filename,2)
        spec.mask = fits.getdata(filename,3)
        spec.sky = fits.getdata(filename,4)
        spec.skyerr = fits.getdata(filename,5)
        spec.telluric = fits.getdata(filename,6)
        spec.telerr = fits.getdata(filename,7)
        #spec.lsf = fits.getdata(filename,8)
        spec.meta = fits.getdata(filename,9)    # meta-data
        # Spectrum, error, sky, skyerr are in units of 1e-17
        #  these are 2D arrays with [Nvisit+2,Npix]
        #  the first two are combined and the rest are the individual spectra
        head1 = fits.getheader(filename,1)
        w0 = head1["CRVAL1"]
        dw = head1["CDELT1"]
        nw = head1["NAXIS1"]
        spec.wave = 10**(np.arange(nw)*dw+w0)
        spec.snr = spec.head["SNR"]
        return spec

    # BOSS spec
    if (base.find("spec-") > -1) | (base.find("SDSS") > -1):
        # HDU1 - binary table of spectral data
        # HDU2 - table with metadata including S/N
        # HDU3 - table with line measurements
        head = fits.getheader(filename,0)
        tab1 = Table.read(filename,1)
        cat1 = Table.read(filename,2)
        spec = Spec1D(tab1["flux"].data[0])
        spec.filename = filename
        spec.sptype = "spec"
        spec.waveregime = "Optical"
        spec.instrument = "BOSS"
        spec.head = head
        # checking for zeros in IVAR
        ivar = tab1["ivar"].data[0].copy()
        bad = (ivar==0)
        if np.sum(bad) > 0:
            ivar[bad] = 1.0
            err = 1.0/np.sqrt(ivar)
            err[bad] = np.nan
        else:
            err = 1.0/np.sqrt(ivar)
        spec.err = err
        spec.ivar = tab1["ivar"].data[0]
        spec.wave = 10**tab1["loglam"].data[0]
        spec.mask = tab1["or_mask"].data[0]
        spec.and_mask = tab1["and_mask"].data[0]
        spec.or_mask = tab1["or_mask"].data[0]
        spec.sky = tab1["sky"].data[0]
        spec.wdisp = tab1["wdisp"].data[0]
        spec.model = tab1["model"].data[0]
        spec.meta = cat1
        # What are the units?
        spec.snr = cat1["SN_MEDIAN_ALL"].data
        return spec        


    # MaStar spec
    if (base.find("mastar-") > -1):
        # HDU1 - table with spectrum and metadata
        tab = Table.read(filename,1)
        spec = Spec1D(tab["FLUX"].data[0])
        spec.filename = filename
        spec.sptype = "MaStar"
        spec.waveregime = "Optical"
        spec.instrument = "BOSS"
        # checking for zeros in IVAR
        ivar = tab["IVAR"].data[0].copy()
        bad = (ivar==0)
        if np.sum(bad) > 0:
            ivar[bad] = 1.0
            err = 1.0/np.sqrt(ivar)
            err[bad] = np.nan
        else:
            err = 1.0/np.sqrt(ivar)
        spec.err = err
        spec.ivar = tab["IVAR"].data[0]
        spec.wave = tab["WAVE"].data[0]
        spec.mask = tab["MASK"].data[0]
        spec.disp = tab["DISP"].data[0]
        spec.presdisp = tab["PREDISP"].data[0]
        meta = {'DRPVER':tab["DRPVER"].data,'MPROCVER':tab["MPROCVER"].data,'MANGAID':tab["MANGAID"].data,'PLATE':tab["PLATE"].data,
                'IFUDESIGN':tab["IFUDESIGN"].data,'MJD':tab["MJD"].data,'IFURA':tab["IFURA"].data,'IFUDEC':tab["IFUDEC"].data,'OBJRA':tab["OBJRA"].data,
                'OBJDEC':tab["OBJDEC"].data,'PSFMAG':tab["PSFMAG"].data,'MNGTARG2':tab["MNGTARG2"].data,'NEXP':tab["NEXP"].data,'HELIOV':tab["HELIOV"].data,
                'VERR':tab["VERR"].data,'V_ERRCODE':tab["V_ERRCODE"].data,'MJDQUAL':tab["MJDQUAL"].data,'SNR':tab["SNR"].data,'PARS':tab["PARS"].data,
                'PARERR':tab["PARERR"].data}
        spec.meta = meta
        # What are the units?
        spec.snr = tab["SNR"].data
        return spec        


    # lp*fits, synthetic spectra
    if base.find("lp") > -1:
        tab = Table.read(filename,format='fits')
        spec = Spec1D(tab["FLUX"].data)
        spec.filename = filename
        spec.sptype = "synthetic"
        spec.wave = tab["WAVE"].data
        spec.mask = np.zeros(len(spec.wave))
        if np.min(spec.wave) < 1e4:
            spec.waveregime = "Optical"
        else:
            spec.waveregime = "NIR"
        spec.instrument = "synthetic"
        # Parse the filename
        # lp0000_XXXXX_YYYY_MMMM_NNNN_Vsini_ZZZZ_SNRKKKK.asc where
        # XXXX gives Teff in K, YYYY is logg in dex (i4.4 format,
        # YYYY*0.01 will give logg in dex),
        # MMMM stands for microturbulent velocity (i4.4 format, fixed at 2.0 km/s),
        # NNNN is macroturbulent velocity (i4.4 format, fixed at 0 km/s),
        # ZZZZ gives projected rotational velocity in km/s (i4.4 format),
        # KKKK refers to SNR (i4.4 format). lp0000 stands for the solar metallicity.
        dum = base.split("_")
        spec.meta = {'Teff':np.float(dum[1]), 'logg':np.float(dum[2])*0.01, 'micro':np.float(dum[3]),
                     'macro':np.float(dum[4]), 'vsini':np.float(dum[6]), 'SNR':np.float(dum[7][3:])}
        spec.snr = spec.meta['SNR']
        spec.err = spec.flux*0.0+1.0/spec.snr
        return spec        
