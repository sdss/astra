import sys, os
#from SDSS import rdspec, Spec1D
#from Network import Network
#from Fit import Fit
#from UncertFit import UncertFit
import numpy as np
import matplotlib.pyplot as plt
#from common import param_names, param_units
#from fit_common import save_figure
from multiprocessing import Pool, Lock

from astra.contrib.zetapayne.SDSS import rdspec, Spec1D
from astra.contrib.zetapayne.Network import Network
from astra.contrib.zetapayne.Fit import Fit
from astra.contrib.zetapayne.UncertFit import UncertFit
from astra.contrib.zetapayne.common import param_names, param_units
from astra.contrib.zetapayne.fit_common import safe_figure

lock = Lock()

def fit_APOGEE(path, NN, Cheb_order):

    spec = rdspec(path)

    wave_ = spec.wave
    
    if len(spec.flux.shape)==2:
        flux_ = spec.flux[0,:]
        err_ = spec.err[0,:]
    else:
        flux_ = spec.flux
        err_ = spec.err

    wave, flux, err = [],[],[]
    for i,v in enumerate(flux_):
        if v != 0.0:
            wave.append(wave_[i])
            flux.append(v)
            err.append(err_[i])

    flux_mean = np.mean(flux)
    flux /= flux_mean
    err /= flux_mean

    fit = Fit(NN, Cheb_order)
    fit.N_presearch_iter = 1
    fit.N_pre_search = 4000
    unc_fit = UncertFit(fit, 22500)
    fit_res = unc_fit.run(wave, flux, err)

    objid = spec.head['OBJID']
    row = [objid]
    k = 0
    for i,v in enumerate(param_names):
        if NN.grid[v][0]!=NN.grid[v][1]:
            row.append('%.2f'%fit_res.popt[k])
            row.append('%.4f'%fit_res.uncert[k])
            k += 1
    row.append('%.2f'%fit_res.popt[k])
    row.append('%.2f'%fit_res.RV_uncert)
    txt = ' '.join(row)

    lock.acquire()
    with open('LOG_APOGEE', 'a') as f:
        f.write(txt)
        f.write('\n')
    print(txt)
    lock.release()

    fit_res.wave = wave
    fit_res.model *= flux_mean
    return fit_res


if __name__=='__main__':
    if len(sys.argv)<2:
        print('Use:', sys.argv[0], '<path to spectrum or folder with spectra>')
        exit()

    input_path = sys.argv[1]

    NN_path = '/home/elwood/Documents/SDSS/NN/APOGEE/G4500_NN_n400_b1000_v0.1.npz'
    NN = Network()
    NN.read_in(NN_path)

    def process_file(fn):
        path = os.path.join(input_path, fn)
        res = fit_APOGEE(path, NN, 10)
        M = np.vstack([res.wave, res.model])
        np.save(path + '.mod', M)
        return 1

    if os.path.isdir(input_path):
        files = [fn for fn in os.listdir(input_path) if fn.endswith('.fits')]
        with Pool() as pool:
            pool.map(process_file, files)
    else:
        process_file(input_path)














