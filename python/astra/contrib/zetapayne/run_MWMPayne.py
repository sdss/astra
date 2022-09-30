import os, sys
import numpy as np
from astra.contrib.zetapayne.bisect import bisect
from astra.contrib.zetapayne.Network import Network
from astra.contrib.zetapayne.common import param_names, param_units, parse_inp
from astra.contrib.zetapayne.Fit import Fit
from astra.contrib.zetapayne.fit_common import save_figure
from astra.contrib.zetapayne.UncertFit import UncertFit
from astra.contrib.zetapayne.random_grid_common import parse_inp
from multiprocessing import Pool, Lock, cpu_count
from astra.contrib.zetapayne.FitLogger import FitLoggerDB
import matplotlib.pyplot as plt
from astra.contrib.zetapayne.SpectrumLoader import SpectrumLoader
from astra.contrib.zetapayne.LSF import *
from astra.contrib.zetapayne.DER_SNR import DER_SNR
import traceback
import datetime

from astra.contrib.zetapayne.astropy.nddata import StdDevUncertainty


lock = Lock()

def refraction_index_V2A(lambda_vacuum):
    """
    Returns refraction index of air for vacuum-air conversion
    Donald Morton (2000, ApJ. Suppl., 130, 403)
    see http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    lambda_vacuum: wavelength in angstrom
    """
    s = 1.e4/lambda_vacuum
    n = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)
    return n

def vacuum_to_air(wave_AA):
    """
    Important: wavelength must be in Angstroems
    """
    wave_new = []
    for w in wave_AA:
        n = refraction_index_V2A(w)
        wave_new.append( w/n )
    return np.array(wave_new)


def fit_spectrum(spectrum, NN, opt, logger, constraints={}):

    wave_start = float(opt['wave_range'][0])
    wave_end = float(opt['wave_range'][1])
    Cheb_order = int(opt['N_chebyshev'])

    #wave = spectrum.wave
    #flux = spectrum.flux
    #err = spectrum.err
    wave = spectrum.wavelength.value
    if spectrum.flux.ndim > 1 and spectrum.flux.shape[0] > 1:
        raise NotImplementedError("Can only handle single spectra at the moment")


    flux = spectrum.flux.value.flatten()
    err = spectrum.uncertainty.represent_as(StdDevUncertainty).array.flatten()

    # This is the logic that was in SpectrumLoader
    good = np.isfinite(flux) * np.isfinite(err) * (err > 0)
    wave = wave[good]
    flux = flux[good]
    err = err[good]


    start_idx = bisect(wave, wave_start)
    end_idx = bisect(wave, wave_end)
    wave = wave[start_idx:end_idx]
    flux = flux[start_idx:end_idx]
    err = err[start_idx:end_idx]
    f_mean = np.mean(flux)
    flux /= f_mean
    err /= f_mean

    wave = vacuum_to_air(wave)

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

    try:
        wresl = spectrum.meta["wresl"].flatten()
        wresl = wresl[good]
    except:
        print(f"Unable to get WRESL, defaulting to SPECTRAL_R")
        fit.lsf = LSF_Fixed_R(float(opt['spectral_R']), wave, NN.wave)
    else:
        fit.lsf = LSF_wave_delta(wresl[start_idx:end_idx], wave, NN.wave)

    fit.N_presearch_iter = int(opt['N_presearch_iter'])
    fit.N_pre_search = int(opt['N_presearch'])
    unc_fit = UncertFit(fit, float(opt['spectral_R']))
    fit_res = unc_fit.run(wave, flux, err)
    CHI2 = fit_res.chi2_func(fit_res.popt)

    #SNR = DER_SNR(flux)
    snr = spectrum.meta["SNR"][0]

    #name = spectrum.obj_id
    #row = [name, '%.1f'%SNR]
    k = 0
    db_values = []
    for i,v in enumerate(param_names):
        if NN.grid[v][0]!=NN.grid[v][1]:
            #row.append('%.2f'%fit_res.popt[k])
            #row.append('%.4f'%fit_res.uncert[k])
            db_values.append(fit_res.popt[k])
            db_values.append(fit_res.uncert[k])
            k += 1
        else:
            db_values.append(np.nan)
            db_values.append(np.nan)

    db_values.append(fit_res.RV)
    db_values.append(fit_res.RV_uncert)
    db_cheb = fit_res.popt[k+1:]

    #row.append('%.2f'%fit_res.RV)
    #row.append('%.2f'%fit_res.RV_uncert)
    #txt = ' '.join(row)

    fit_res.wave = wave
    fit_res.model *= f_mean

    name = str(spectrum.meta.get("CAT_ID", spectrum.meta.get("SDSS_ID", "UNKNOWN")))
    print(f"Name given is {name}, check this later")

    try:
        logger.add_record(name, snr, db_values, db_cheb)
        #logger.add_metadata(spectrum.full_path, spectrum.ra_dec, [wave, flux*f_mean, fit_res.model])
        logger.save_plot(wave, flux*f_mean, fit_res.model, name)
        logger.save_RV_P_plot(fit_res.RV_P_plot[0], fit_res.RV_P_plot[1], name)
    except:
        print(f"Plots failed, .. continuing ")
        

    keys = ('teff', 'e_teff', 'logg', 'e_logg', 'vsini', 'e_vsini', 'v_micro', 'e_v_micro', 'fe_h', 'e_fe_h', 'v_rel', 'e_v_rel')
    result = dict(zip(keys, db_values))
    result.update(
        theta=db_cheb,
        snr=snr,
        chi_sq=CHI2,
        reduced_chi_sq=CHI2/(flux.size - len(db_values) - 1)
    )
    # Create a model spectrum
    # Resample model flux back to observed spectrum
    model_flux = np.interp(
        spectrum.wavelength.value,
        fit_res.wave,
        fit_res.model,
        left=np.nan,
        right=np.nan
    )

    # Put the result in the format as if we had many results.
    meta = dict(model_flux=model_flux)

    return ([result], [meta], [fit_res])


if __name__=='__main__':
    if len(sys.argv)<2:
        print('Use:', sys.argv[0], '<config file>')
        exit()

    opt = parse_inp(sys.argv[1])

    NN_path = opt['NN_path'][0]
    NN = Network()
    NN.read_in(NN_path)

    constr = {}
    #constr['[M/H]'] = (0.2-0.01, 0.2+0.01)

    logger = FitLoggerDB(opt['log_dir'][0])
    logger.init_DB()
    logger.new_run(str(opt))

    loader = SpectrumLoader(opt['data_format'][0])
    stop_code = 'LLCQHW'

    def process(sp):
        try:
            if os.path.isfile(stop_code): return
            sd = sp.load()
            res = fit_BOSS(sd, NN, opt, logger)
        except:
            exc_text = traceback.format_exc()
            date = str(datetime.datetime.now())
            with open('ERROR_LOG', 'a') as errlog:
                errlog.write('-'*50 + '\n')
                errlog.write(date+'\n')
                errlog.write(exc_text+'\n')
            print(exc_text)

    regex = '.'
    if 'name_regex' in opt:
        regex = opt['name_regex'][0]

    spectra = loader.get_spectra(opt['data_path'][0], regex)
    parallel = opt['parallel'][0].lower() in ['true', 'yes', '1']
    if 'N_threads' in opt:
        N_threads = int(opt['N_threads'][0])
    else:
        N_threads = cpu_count()//2

    if parallel:
        print('Parallel processing option is ON, running with ' + str(N_threads) + ' threads')
        with Pool(processes=N_threads) as pool:
            pool.map(process, spectra)
    else:
        for sp in spectra: process(sp)

    logger.close()

    if os.path.isfile(stop_code):
        os.remove(stop_code)





