import sys, gc
import os
from math import isnan
import numpy as np
from random_grid_common import *


if len(sys.argv) < 2:
    print('Usage:', sys.argv[0], '<path to the folder with models> [N models to assemble]')
    exit()

path = sys.argv[1]

N_limit = -1
if len(sys.argv) > 2:
    N_limit = int(sys.argv[2])
    if N_limit <= 0:
        print('Zero models to assemble, exiting.')
        exit()

grid_fn = '_GRID.npz'
if N_limit > 0:
    grid_fn = '_GRID_' + str(N_limit) + '.npz'

files = [fn for fn in os.listdir(path) if fn.endswith('.npz') and not fn.startswith('_')]
files.sort()

fluxes, params = [],[]

N = 0
for fn in files:
    N += 1
    if N_limit > 0 and N > N_limit: break
    npz = np.load(os.path.join(path, fn), allow_pickle=True)
    flx = np.squeeze(npz['flux'])
    flx_sum = sum(flx)
    if isnan(flx_sum):
        print('NANs in flux: excluded from grid')
        continue
    fluxes.append(flx)
    param_dict = npz['labels'].item()
    pp = []
    for p in param_names:
        if p in param_dict:
            pp.append(param_dict[p])
    params.append(pp)
    print(fn)

opt_path = os.path.join(path, '_grid.conf')
opt = parse_inp(opt_path)
wave = [float(x) for x in opt['wavelength']]
wave_grid = np.linspace(wave[0], wave[1], len(fluxes[0]))

grid = {}
for o in opt:
    if o in param_names:
        grid[o] = [float(x) for x in opt[o]]

np.savez(os.path.join(path, grid_fn), flux=fluxes, labels=params, wvl=wave_grid, grid=grid)

print('Assembled grid saved to', os.path.join(path, grid_fn))



