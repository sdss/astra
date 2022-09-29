import sys, shutil, os
import subprocess as sp


def run_GSSP_grid(run_id, output_path, parameters, wave_range, GSSP_cmd, base_data_path, scratch_dir, resolution='-1', Kurucz=True):
    """
    This routine runs GSSP in grid mode.
    ------------------------------------
    run_id: unique id to prevent conflict when running multiple instances
    output_path: where to save the generated inp file
    parameters: dictionary with keys ['T_eff','log(g)','v*sin(i)','v_micro','[M/H]']
                every entry is a list/tuple [start, end, step]
                example: parameters['T_eff'] = [5000, 9000, 100]
    wave_range: tuple with wavelength range and step (start, end, step)
    GSSP_cmd: command to run GSSP, such as './GSSP' for single thread
    base_data_path: path where 'abundances', 'Kurucz'/'LLmodels' and 'Radiative_transfer'
                are located
    resolution: spectral resolving power
    Kurucz: use Kurucz models if True, otherwise LLmodels
    """

    def s(L, ns):
        fmt = '%.' + str(ns) + 'f'
        return [fmt%x for x in L]

    Teff = s(parameters['T_eff'], 0)
    logg = s(parameters['log(g)'], 1)
    vsini = s(parameters['v*sin(i)'], 0)
    vmicro = s(parameters['v_micro'], 1)
    metal = s(parameters['[M/H]'], 1)
    wave = s(wave_range, 4)

    with open(output_path, 'w') as f:
        f.write(str(run_id) + '\n')
        f.write(scratch_dir)
        if scratch_dir[-1]!='/': f.write('/')
        f.write('\n')
        f.write(' '.join([Teff[0], Teff[2], Teff[1]]) + '\n' )
        f.write(' '.join([logg[0], logg[2], logg[1]]) + '\n' )
        f.write(' '.join([vmicro[0], vmicro[2], vmicro[1]]) + '\n' )
        f.write(' '.join([vsini[0], vsini[2], vsini[1]]) + '\n' )
        f.write('skip 0.03 0.02 0.07\n')
        f.write(' '.join(['skip', metal[0], metal[2], metal[1]]) + '\n' )
        f.write('He 0.04 0.005 0.06\n')
        f.write('0.0 1.0 0.0 '+resolution+'\n')
        f.write(os.path.join(base_data_path, 'abundances') + '/\n')
        if Kurucz:
            f.write(os.path.join(base_data_path, 'Kurucz') + '/\n')
        else:
            f.write(os.path.join(base_data_path, 'LLmodels') + '/\n')
        f.write(os.path.join(base_data_path, 'Radiative_transfer/VALD2012.lns') + '\n')
        f.write('2 1\n')
        f.write('ST\n')
        f.write('1 '+wave[2]+' grid\n')
        f.write('input_data.norm\n')
        f.write('0.5 0.99 5.9295 adjust\n')
        f.write(wave[0]+' '+wave[1]+'\n')
        
    shutil.rmtree(os.path.join('rgs_files', str(run_id)), ignore_errors=True)
    
    ok = True
    log_fn = output_path+'.log'
    try:
        o = sp.check_output(GSSP_cmd + ' ' + output_path, shell=True, stderr=sp.STDOUT)
    except sp.CalledProcessError as err:
        print(err.output)
        with open(log_fn, 'wb') as f: f.write(err.output)
        ok = False
    else:
        with open(log_fn, 'wb') as f: f.write(o)

    return ok




