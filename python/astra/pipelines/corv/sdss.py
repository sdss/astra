#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 09:06:32 2021

Functions specific to SDSS-5, including data processing. 

@author: vedantchandra
"""

# SPIN OFF THESE PATHS TO SWITCH BASED ON WHETHER LAPTOP OR HOLY

import numpy as np
from astropy.io import fits
import astropy
import glob
from astropy.table import Table
from tqdm import tqdm
import socket
hostname = socket.gethostname()

from astra.pipelines.corv import spectral_resampling
from astra.pipelines.corv import utils

# if hostname[:4] == 'holy':
# 	print('using holyoke paths')
# 	datapath = '/n/holyscratch01/conroy_lab/vchandra/wd/6_0_4/' # abs. path with CATID folders
# 	catpath = '/n/home03/vchandra/wd/01_ddwds/cat/' # abs. path with CATID folders
# else:
# 	print('using local paths')
# 	datapath = '/Users/vedantchandra/0_research/01_sdss5/006_build_corv/data/ddcands/' # abs. path with CATID folders
# 	catpath = '/Users/vedantchandra/0_research/01_sdss5/006_build_corv/cat/' # abs. path with CATID folders

## FOR COMMISSIONING

datapath = '/Users/vedantchandra/0_research/01_sdss5/006_build_corv/data/comm/'
catpath = '/Users/vedantchandra/0_research/01_sdss5/006_build_corv/data/comm_cat/'

print(catpath)

try:
	starcat = Table.read(catpath + 'starcat.fits')
	expcat = Table.read(catpath + 'expcat.fits')
	#print('starcat has %i stars' % len(starcat))
except:
	print('star and exposure catalogs not found! check paths and run make_catalogs() if you want to use sdss functionality. otherwise ignore.')


####### I/O FUNCTIONS #############

# Construct exp and exps data structures: dictionaries that contain all the required data

# From SDSS-V co-add datamodel
# There are the final wavelengths the co-adds are interpolated on

loglam0 = 3.55230
step = 0.0001
nwl = 4648
loglamgrid = np.linspace(loglam0, loglam0 + nwl * step, nwl)
lamgrid = 10**loglamgrid
break_wl = 5900
log_break_wl = np.log10(break_wl)

import lmfit

# ADD HELIUM

default_centres =  dict(ha = 6564.61, hb = 4862.68, hg = 4341.68, hd = 4102.89,
                 he = 3971.20, hz = 3890.12, hn = 3835.5,
             ht = 3799.5)
default_windows = dict(ha = 100, hb = 100, hg = 85, hd = 70, he = 30,
                  hz = 25, hn = 15, ht = 10)
default_edges = dict(ha = 25, hb = 25, hg = 20, hd = 20, 
                he = 5, hz = 5, hn = 5, ht = 4)

default_names = ['hn', 'hz', 'he', 'hd', 'hg', 'hb', 'ha']

def get_ew_line(wl, fl, ivar, line, window = 150, edge = 20, plot = False):
    
    cwl, cfl, civar = utils.cont_norm_line(wl, fl, ivar, line, window, edge)

    model = lmfit.models.ConstantModel() - lmfit.models.VoigtModel(prefix = '')

    params = model.make_params()
    params['center'].set(value = line)
    params['c'].set(value = 1, vary = False)
    params['sigma'].set(value = 10)
    params['amplitude'].set(value = 10)

    try:

    	res = model.fit(cfl, params = params, x = cwl,
                   weights = np.sqrt(civar))
    except:
        print('fit failed, returning NaN')
        return np.nan, np.nan, np.nan
    
    modfl = model.eval(res.params, x = cwl)
    
    if plot:

        plt.plot(cwl, cfl, 'k.', label = 'Data')
        plt.plot(cwl, model.eval(params, x = cwl))

        plt.plot(cwl, modfl, 'r', label = 'Gaussian Model')

        plt.legend()

        plt.ylabel('Normalized Flux')
        plt.xlabel(r"$\lambda$ ($\AA$)")
        plt.show()
        
    diff = np.diff(cwl)
    dx = np.abs(np.concatenate((diff, [diff[-1]])))
    ew = np.sum((1 - modfl / 1) * dx)
    
    
    return ew, res.params['fwhm'].value, res.params['height'].value

def get_ew_lines(wl, fl, ivar, names = default_names, 
                 centres = default_centres, windows = default_windows, edges = default_edges, 
                 plot = False):
    
    ret = {};
    for name in names:
        ew, fwhm, height = get_ew_line(wl, fl, ivar, centres[name], windows[name], edges[name], plot = plot)
        ret[name + '_ew'] = ew
        ret[name + '_fwhm'] = fwhm
        ret[name + '_height'] = height
    return ret

def get_ew_dict(cid):
    exps = get_exposures(cid)

    wl, fl, ivar = make_coadd(exps)

    ew_dict = get_ew_lines(wl, fl, ivar, plot = False)
    ew_dict['cid'] = cid
    
    return ew_dict

def make_dacat(min_ew = 3, max_ew = 50):

	ewcat = Table.read(catpath + 'ewcat.fits')

	DA = np.repeat(True, len(ewcat))

	names = np.flip(['hd', 'hg', 'hb', 'ha'])

	for name in names:
	    DA = DA & ((ewcat[name + '_ew'] > min_ew) & (ewcat[name + '_ew'] < max_ew))

    print('There are %i stars' % len(DA))
	print('Of these, %i seem to be DAs' % np.sum(DA))

	ewcat[DA].write(catpath + 'dacat.fits', overwrite = True)

	# print(ewcat[~DA])

	print('written %sdacat.fits' % catpath)



def get_exposure(bf, rf):
	
	# Given a red and blue single exposure .fits file, return `exp` object
	
	exp = {};

	with fits.open(bf) as f:
		
		exp['header'] = dict(f[0].header)
		
		bwl = f[3].data
		bfl = f[0].data
		bivar = f[1].data
		bdisp = f[4].data
		bsky = f[5].data
		bmask = f[2].data
		
	with fits.open(rf) as f:
		
		rwl = f[3].data
		rfl = f[0].data
		rmask = f[2].data
		rivar = f[1].data
		rdisp = f[4].data
		rsky = f[5].data
		
	rsel = rwl > log_break_wl
	bsel = bwl < log_break_wl
		
	exp['logwl'] = np.concatenate((bwl[bsel], rwl[rsel])).astype(float)
	exp['fl'] = np.concatenate((bfl[bsel], rfl[rsel])).astype(float)
	exp['ivar'] = np.concatenate((bivar[bsel], rivar[rsel])).astype(float)
	exp['wdisp'] = np.concatenate((bdisp[bsel], rdisp[rsel])).astype(float)
	exp['sky'] = np.concatenate((bsky[bsel], rsky[rsel])).astype(float)
	exp['mask'] = np.concatenate((bmask[bsel], rmask[rsel])).astype(float)
		
	return exp

def get_exposures(catalogid):
	
	# Given a catalogid, get all exposures. Return exp data and table
	
	seltable = expcat[expcat['cid'] == catalogid]
	expsdata = [];
	expsheader = [];
	
	seltable = seltable[np.argsort(seltable['TAI-BEG'])]
	
	for ii,row in enumerate(seltable):
		exp = get_exposure(row['bfile'], row['rfile'])
		exp.pop('header') # no need to duplicate what's in the table already
		expsdata.append(exp)
			
	exps = dict(data = expsdata, header = seltable)
	
	return exps

def make_coadd(exps, method = 'ivar_mean'):
	
	# Todo: add WDISP here too
	# Todo: some way to pass velocities here to shift and then co-add
	# Todo: add different coadding methods: median, inverse variance mean. 
	
	nexp = len(exps['data'])

	fls = np.zeros((nexp, nwl))
	sigmas = np.zeros((nexp, nwl))

	for ii,exp in enumerate(exps['data']):

		with np.errstate(divide='ignore'):
			fl_i, sigma_i = spectral_resampling.spectres(loglamgrid, exp['logwl'], exp['fl'], np.reciprocal(np.sqrt(exp['ivar'])))

		fls[ii, :] = fl_i
		sigmas[ii, :] = sigma_i

	fl = np.zeros(nwl)
	sigma = np.zeros(nwl)
	mask = np.ones(nwl).astype(bool)
	#nexps = np.zeros(nwl)

	for px in range(nwl):
				
		fl_px = fls[:, px]
		sigma_px = sigmas[:, px]
		
		# sigma clipping? disabled for now.. 
		# fl_px_clipped, lb, ub = stats.sigmaclip(fl_px, low = 5.0, high = 5.0)
		# sigma_px = sigma_px[(fl_px > lb) & (fl_px < ub)]
		# fl_px = fl_px_clipped
						
		goodmask = np.isfinite(sigma_px)
		goodmask *= (fl_px > 0)
		ngood = np.sum(goodmask)
		#nexps[px] = ngood

		if ngood > 0:

			if method == 'median':
				fl[px] = np.median(fl_px[goodmask])
				sigma[px] = np.sqrt(np.sum(sigma_px[goodmask]**2)) / ngood # CHECK MATH HERE, ERROR ON MEDIAN? 
			elif method == 'mean':
				fl[px] = np.mean(fl_px[goodmask])
				sigma[px] = np.sqrt(np.sum(sigma_px[goodmask]**2)) / ngood
			elif method == 'ivar_mean':
				fl[px] = np.average(fl_px[goodmask], weights = 1/sigma_px[goodmask]**2)
				sigma[px] = np.sqrt(1 / np.sum(1/sigma_px[goodmask]**2))
			else:
				print('invalid method!')

		else:
			mask[px] = False # where mask == False, there is zero usable data
			fl[px] = np.nan
			sigma[px] = np.inf
			
	ivar = 1 / sigma**2
	ivar[~mask] = 0.0
	
	return lamgrid, fl, ivar

####### CATALOG CONSTRUCTION #############


def make_catalogs(make_filecat = True, make_expcat = True, make_starcat = True,
	catpath = catpath, datapath = datapath):

	# Make filecat, expcat, and starcat given the datapath and output catpath. 

	#### Make `filecat.fits`

	if make_filecat:

		print('making %sfilecat.fits' % catpath)

		fits_files = glob.glob(datapath + '/*/*.fits')
		filecat_rows = []; 

		for nn,file in enumerate(tqdm(fits_files)):
			
			with fits.open(file) as f:
			
				row = dict(f[0].header)
				row['filepath'] = file
				row['cid'] = int(file.split('-')[-3])
				row['camera'] = file.split('-')[-2]
				row['expid'] = int(file.split('-')[-1].split('.')[0])
				
			filecat_rows.append(row)
			

		filecat = Table(filecat_rows)
		print('filecat has %i rows' % len(filecat))
		filecat.write(catpath + 'filecat.fits', overwrite = True)

	#### Make `expcat.fits`

	if make_expcat:

		filecat = Table.read(catpath + 'filecat.fits')

		mjds = np.unique(filecat['MJD'])

		expcat_rows = [];

		for mjd in tqdm(mjds):

		    mjdtable = filecat[filecat['MJD'] == mjd]
		    
		    cids = np.unique(mjdtable['cid'])
		    
		    for cid in cids:
		        
		        cidtable = mjdtable[mjdtable['cid'] == cid]

		        expids = np.unique(cidtable['expid'])

		        for expid in (expids):

		            seltab = cidtable[cidtable['expid'] == expid]
		            row = dict(seltab[0])
		            row.pop('camera')
		            row.pop('filepath')

		            redfile = seltab[seltab['camera'] == 'r1']['filepath'][0]
		            bluefile = seltab[seltab['camera'] == 'b1']['filepath'][0]

		            row['rfile'] = redfile
		            row['bfile'] = bluefile

		            row['dr2_id'] = int(seltab['G_DR2'][0])

		            expcat_rows.append(row)

		expcat = Table(expcat_rows)
		print('expcat has %i rows' % len(expcat))
		expcat.write(catpath + 'expcat.fits', overwrite = True)

	#### Make `starcat.fits`

	if make_starcat:

		expcat = Table.read(catpath + 'expcat.fits')

		print('making %sstarcat.fits' % catpath)

		cids = np.unique(expcat['cid'])
		starcat_rows = [];

		for cid in tqdm(cids):
			seltab = expcat[expcat['cid'] == cid]
			
			row = {}
			row['cid'] = cid
			row['nexp'] = len(seltab)
			row['dr2_id'] = seltab['dr2_id'][0]
			row['spec_ra'] = np.round(np.median(seltab['RA']), 5)
			row['spec_dec'] = np.round(np.median(seltab['DEC']), 5)
			
			starcat_rows.append(row)

		starcat = Table(starcat_rows)
		print('starcat has %i rows' % len(starcat))
		starcat.write(catpath + 'starcat.fits', overwrite = True)
