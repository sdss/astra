import sys, shutil
import os, time
import numpy as np
from random_grid_common import *
import sobol_seq
from scipy.interpolate import interp1d
from shutil import copyfile
from CustomPool import CustomPool

opt_fn = sys.argv[1]
opt = parse_inp(opt_fn)

N_models = int(opt['N_models_to_sample'][0])
N_models_skip = int(opt['N_models_to_skip'][0])
N_instances = int(opt['N_instances'][0])

rnd_grid_dir = opt['output_dir'][0]
subgrid_dir = 'subgrids'
for dn in [rnd_grid_dir, subgrid_dir]:
    if not os.path.exists(dn):
        os.makedirs(dn)

copyfile(opt_fn, os.path.join(rnd_grid_dir, '_grid.conf'))

grid = {}
for o in opt:
    if o in param_names:
        grid[o] = [float(x) for x in opt[o]]

###############################################################
#       Example of how to generate a quasi-random grid
###############################################################

grid_params = [p for p in param_names if grid[p][0]!=grid[p][1]]

# --- Specify number of free parameters and total grid points -
N_param = len(grid_params)

# --- Calculate Sobol numbers for random sampling -------------
# This creates a N_grid x N_param matrix of quasi-random numbers
# between 0 and 1 using Sobol numbers
N_sobol = sobol_seq.i4_sobol_generate(N_param, N_models)
print()
print('------ N_grid x N_param matrix of Sobol numbers -------')
print(N_sobol)
print()

# --- Prepare linear interpolation functions for mapping ------
# One such function is needed for every free paraneter, i.e. 
# a total of N_param functions

# Range of the Sobol numbers
intrp_range = [0, 1]       

# 1D linear interpolation functions
intrp = [interp1d(intrp_range, grid[p][:2]) for p in grid_params]

# --- Create final quasi-random sampled grid -----------------
# We make a new matrix theta, where each column corresponds 
# to one of the free parameters, whose ranges have been mapped
# onto the Sobol numbers

columns = [v(N_sobol[:,i]) for i,v in enumerate(intrp)]
theta = np.vstack(columns).T
print()
print('------ N_grid x N_param quasi-random grid -------')
print(theta)
print()

np.savetxt('theta.data', theta)

work = []
for i in range(N_models):
    if i < N_models_skip: continue
    pp_arr = theta[i,:]
    pp = {}
    for j,v in enumerate(grid_params):
        pp[v] = pp_arr[j]
    subgrid = create_subgrid(pp, grid)
    work_item = (str(i).zfill(6), subgrid, pp, pp_arr, subgrid_dir, opt)
    work.append(work_item)
    

if N_instances>1:
    with CustomPool(processes=N_instances) as pool:
        ret = pool.map(run_one_item, work, chunksize=1)
else:
    for item in work: run_one_item(item)


print('Done.')










