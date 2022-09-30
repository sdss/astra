import os, time
import numpy as np
#from Partition import *
#from run_GSSP import *
#from Grid import *
#from RandomGrid import *
#from Subgrid import *
#from random_grid_common import *

from astra.contrib.zetapayne.Partition import *
from astra.contrib.zetapayne.run_GSSP import *
from astra.contrib.zetapayne.Grid import *
from astra.contrib.zetapayne.RandomGrid import *
from astra.contrib.zetapayne.Subgrid import *
from astra.contrib.zetapayne.random_grid_common import *
 

opt = parse_inp()

if DEBUG:
    for o in opt: print(o, opt[o])

wave = [float(x) for x in opt['wavelength']]
N_wave_points = int( (wave[1]-wave[0])/wave[2] )
N_bytes_per_model = 8 * N_wave_points

memory_limit = float(opt['memory_limit_GB'][0]) * 1.e9

models_limit = memory_limit / N_bytes_per_model

grid = {}
for o in opt:
    if o in param_names:
        grid[o] = [float(x) for x in opt[o]]

if DEBUG:
    print('-'*25)
    print('Stellar parameters:', grid)

print('-'*25)
PART = Partition(grid, models_limit)
PART.optimize_partition()
PART.check()

if not os.path.exists(rnd_grid_dir):
    os.makedirs(rnd_grid_dir)

print('-'*25)
sample_density = float(opt['N_models_to_sample'][0]) / PART.total_volume
print('Sample density:', sample_density)

SUB = Subgrid(grid, PART, param_names)
n_sub = 0
while SUB.next_subgrid():

    print('-'*25)
    print('Current subgrid:')
    for p in SUB.subgrid: print(p, SUB.subgrid[p])
    print('Volume:', SUB.volume())
    n_sub += 1
    
    subgrid_samples = int(SUB.volume() * sample_density)
    assert subgrid_samples > 0

    print('Subgrid samples:', subgrid_samples)

    if DEBUG: continue

    GSSP_run_cmd = opt['GSSP_run_cmd'][0]
    print('>>>> RUNNING GSSP')
    run_GSSP_grid('subgrid.inp', SUB.subgrid, wave, GSSP_run_cmd)
    print('>>>> GSSP FINISHED')

    print('Loading subgrid')
    GRID = Grid('rgs_files', '.')
    GRID.load()
    print('Subgrid loaded, sampling random grid')

    prefix = '%.0f'%(time.time()*1.e6)

    RND = RandomGrid(GRID)
    for i in range(subgrid_samples):
        fn = prefix + '_' + str(i).zfill(8) + '.npz'
        pp, sp = RND.sample_model()
        np.savez(os.path.join(rnd_grid_dir, fn), flux=sp, labels=pp)

print('-'*25)
print('Number of subgrids visited:', n_sub)    

print('Done.')



















