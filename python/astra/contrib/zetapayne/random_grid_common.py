from common import *
import os
import numpy as np
from run_GSSP import *
from Grid import *
from RandomGrid import *
from shutil import copyfile


DEBUG = True

def run_one_item(item):
    """
    Runs GSSP to create a subgrid
    """
    (run_id, subgrid, pp, pp_arr, subgrid_dir, opt) = item

    wave = [float(x) for x in opt['wavelength']]
    GSSP_run_cmd = opt['GSSP_run_cmd'][0]
    GSSP_data_path = opt['GSSP_data_path'][0]
    N_interpol_threads = int(opt['N_interpol_threads'][0])
    scratch_dir = opt['scratch_dir'][0]

    Kurucz = True
    if 'Kurucz' in opt:
        Kurucz = opt['Kurucz'][0].lower() in ['true', 'yes', '1']

    rnd_grid_dir = opt['output_dir'][0]

    inp_fn = os.path.join(subgrid_dir, 'subgrid_' + run_id + '.inp')

    ok = run_GSSP_grid(run_id, inp_fn, subgrid, wave, GSSP_run_cmd, GSSP_data_path, scratch_dir, opt['R'][0], Kurucz=Kurucz)
    if not ok:
        print('GSSP exited with error, item id '+run_id)
        return 1

    rgs_dir = os.path.join('rgs_files', run_id)
    GRID = Grid(rgs_dir)
    GRID.load()

    RND = RandomGrid(GRID)

    fn = run_id + '.npz'
    sp = RND.interpolate(pp_arr, N_interpol_threads)
    np.savez(os.path.join(rnd_grid_dir, fn), flux=sp, labels=pp, wave=wave)
    shutil.rmtree(rgs_dir, ignore_errors=True)

    print('Grid model '+run_id+' complete')

    return 0


def _create_subgrid(pp, grid):
    """
    Creates a single-cell subgrid
    ---
    pp: dictionary with parameter values
    grid: dictionary with grid boundaries
    ---
    Returns disctionary with subgrid boundaries
    """
    vsini = param_names[2]
    grid_params = [p for p in param_names if grid[p][0]!=grid[p][1]]
    subgrid = {}
    for p in param_names:
        if len(GSSP_steps[p]) == 1:
            step = GSSP_steps[p][0]
        else:
            step = GSSP_steps[p][1] if pp[p]<GSSP_steps[p][0] else GSSP_steps[p][2]

        if p==vsini:
            subgrid[p] = [pp[p], pp[p], step]
        elif p in grid_params:
            start = pp[p] - pp[p]%step
            subgrid[p] = [start, start + step, step]
        else:
            subgrid[p] = grid[p] + [step]

    return subgrid

def create_subgrid(pp, grid):
    """
    Creates a single-cell subgrid
    ---
    pp: dictionary with parameter values
    grid: dictionary with grid boundaries
    ---
    Returns disctionary with subgrid boundaries
    """
    grid_params = [p for p in param_names if grid[p][0]!=grid[p][1]]
    subgrid = {}
    for p in param_names:
        if len(GSSP_steps[p]) == 1:
            step = GSSP_steps[p][0]
        else:
            step = GSSP_steps[p][1] if pp[p]<GSSP_steps[p][0] else GSSP_steps[p][2]

        if p in grid_params:
            start = pp[p] - pp[p]%step
            subgrid[p] = [start, start + step, step]
        else:
            subgrid[p] = grid[p] + [step]

    return subgrid




