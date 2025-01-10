import os
import numpy as np
from typing import Iterable
from time import time
from astra.utils import log, expand_path
from astra.pipelines.ferre.operator import post_execution_interpolation
from astra.pipelines.ferre.utils import (
    read_ferre_headers,
    read_control_file, 
    read_input_parameter_file,
    read_output_parameter_file,
    read_and_sort_output_data_file,
    get_processing_times,
    parse_ferre_spectrum_name,
    parse_header_path,
    TRANSLATE_LABELS
)

def write_pixel_array_with_names(path, names, data):
    os.system(f"mv {path} {path}.original")
    np.savetxt(
        path,
        np.hstack([np.atleast_2d(names).reshape((-1, 1)), data]).astype(str),
        fmt="%s"
    )

LARGE = 1e10 # TODO: This is also defined in pre_process, move it common

def post_process_ferre(input_nml_path, **kwargs) -> list[dict]:

    """
        if relative_mode:
            ref_dir = os.path.dirname(dir)
        else:
            ref_dir = None
        log.info(f"Post-processing FERRE results in {dir} {'with FERRE list mode' if relative_mode else 'in standard mode'}")
        for kwds in post_process_ferre(dir, ref_dir, skip_pixel_arrays=skip_pixel_arrays, **kwargs):
            yield FerreChemicalAbundances(**kwds)    
    """
    is_abundance_mode = input_nml_path.endswith("input_list.nml")
    if is_abundance_mode:
        abundance_dir = os.path.dirname(input_nml_path)
        with open(input_nml_path, "r") as fp:
            dirs = [os.path.join(abundance_dir, line.split("/")[0]) for line in fp.read().strip().split("\n")]
        
        # TODO: this might be slow we can probably use -l mode
        for d in dirs:
            post_execution_interpolation(d)
        
        ref_dir = os.path.dirname(input_nml_path)
        v = [list(_post_process_ferre(d, ref_dir, skip_pixel_arrays=True, **kwargs)) for d in dirs]
    else:
        directory = os.path.dirname(input_nml_path)
        post_execution_interpolation(directory)
        v = list(_post_process_ferre(directory, **kwargs))
    return v

def _post_process_ferre(dir, pwd=None, skip_pixel_arrays=False, **kwargs) -> Iterable[dict]:
    """
    Post-process results from a FERRE execution.

    :param dir:
        The working directory of the FERRE execution.
    
    :param pwd: [optional]
        The directory where FERRE was actually executed from. Normally `pwd` and `dir` will always be
        the same, so this keyword argument is optional. However, if FERRE is run in abundance mode
        with the `-l` flag, it might be executed from a path like `analysis/abundances/GKg_b/` but
        the individual FERRE executions exist in places like:

        `abundances/GKg_b/Al`
        `abundances/GKg_b/Mg`

        In these cases, the thing you want is `post_process_ferre('abundances/GKg_b/Al', 'abundances/GKg_b')`.
    """
    
    absolute_dir = expand_path(dir)
    ref_dir = pwd or absolute_dir 

    # When finding paths, if the path is in the input.nml file, we should use `ref_dir`, otherwise `dir`.
    timing = {}
    """
    try:
        raw_timing = np.atleast_2d(np.loadtxt(os.path.join(ref_dir, "timing.csv"), dtype=str, delimiter=","))
    except:
        log.warning(f"No FERRE timing information available for execution in {ref_dir}")
    else:
        try:                
            for name, relative_input_nml_path, t_load, t_elapsed in raw_timing:
                timing.setdefault(relative_input_nml_path, {})
                timing[relative_input_nml_path][name] = (float(t_load), float(t_elapsed))

            if ref_dir != dir:
                relative_input_nml_path = dir[len(ref_dir) + 1:] + "/input.nml"
                timing = timing[relative_input_nml_path]
            else:
                timing = timing["input.nml"]
        except:
            log.exception(f"Exception when trying to load timing for {ref_dir}")
    """
    control_kwds = read_control_file(os.path.join(dir, "input.nml"))

    # Load input files.
    input_names, input_parameters = read_input_parameter_file(ref_dir, control_kwds)   
    N = len(input_names)

    try:
        parameters, e_parameters, meta, names_with_missing_outputs = read_output_parameter_file(ref_dir, control_kwds, input_names)
    except:
        D = int(control_kwds["NDIM"])
        parameters = np.nan * np.ones((N, D))
        e_parameters = np.ones_like(parameters)
        meta = {
            "log_snr_sq": np.nan * np.ones(N),
            "log_chisq_fit": np.nan * np.ones(N),
        }
        names_with_missing_outputs = input_names

    if len(names_with_missing_outputs) > 0:
        log.warn(f"The following {len(names_with_missing_outputs)} are missing outputs: {names_with_missing_outputs}")

    offile_path = os.path.join(ref_dir, control_kwds["OFFILE"])
    # Load and sort the rectified model flux path because this happens in abundances when we would normally use skip_pixel_arrays=True
    '''
    try:
        rectified_model_flux, names_with_missing_rectified_model_flux, output_rectified_model_flux_indices = read_and_sort_output_data_file(
            offile_path, 
            input_names
        )
        write_pixel_array_with_names(offile_path, input_names, rectified_model_flux)
    except:
        log.exception(f"Exception when trying to read and sort {offile_path}")
        names_with_missing_rectified_model_flux = input_names
        rectified_model_flux = np.nan * np.ones((N, 7514))
        is_missing_rectified_model_flux = np.ones(N, dtype=bool)
    else:
        is_missing_rectified_model_flux = ~np.all(np.isfinite(rectified_model_flux), axis=1)
    '''
    
    parameter_input_path = os.path.join(dir, "parameter.input")
    os.system(f"vaffoff {parameter_input_path} {offile_path}")
    is_missing_rectified_model_flux = ~np.isfinite(np.atleast_1d(np.loadtxt(offile_path, usecols=(1, ), dtype=float)))
    names_with_missing_rectified_model_flux = input_names[is_missing_rectified_model_flux]

    if not skip_pixel_arrays:
        #flux = np.atleast_2d(np.loadtxt(os.path.join(ref_dir, control_kwds["FFILE"])))
        #e_flux = np.atleast_2d(np.loadtxt(os.path.join(ref_dir, control_kwds["ERFILE"])))
                            
        sffile_path = os.path.join(ref_dir, control_kwds["SFFILE"])
        '''
        try:
            rectified_flux, names_with_missing_rectified_flux, output_rectified_flux_indices = read_and_sort_output_data_file(
                sffile_path,
                input_names
            )
            # Re-write the model flux file with the correct names.
            write_pixel_array_with_names(sffile_path, input_names, rectified_flux)
        except:
            log.exception(f"Exception when trying to read and sort {sffile_path}")
            names_with_missing_rectified_flux = input_names
            rectified_flux = np.nan * np.ones_like(flux)
        '''
        
        os.system(f"vaffoff {parameter_input_path} {sffile_path}")
        names_with_missing_rectified_flux = input_names[~np.isfinite(np.loadtxt(sffile_path, usecols=(1, ), dtype=float))]
        
        '''
        model_flux_output_path = os.path.join(absolute_dir, "model_flux.output") # TODO: Should this be ref_dir?
        if os.path.exists(model_flux_output_path):
            
            model_flux, *_ = read_and_sort_output_data_file(
                model_flux_output_path,
                input_names
            )            
            write_pixel_array_with_names(model_flux_output_path, input_names, model_flux)
        else:
            log.warn(f"Cannot find model_flux output in {absolute_dir} ({model_flux_output_path})")
            model_flux = np.nan * np.ones_like(flux)
        '''
        model_flux_output_path = os.path.join(absolute_dir, "model_flux.output") # TODO: Should this be ref_dir?
        # We might have to wait some time for it to be written
        #t_wait = time()
        #while not os.path.exists(model_flux_output_path):
        #    if time() > t_wait + 60:
        #        log.warn(f"Cannot find model_flux output in {absolute_dir} ({model_flux_output_path})")
        #        break

        os.system(f"vaffoff {parameter_input_path} {model_flux_output_path}")
        is_missing_model_flux = ~np.isfinite(np.atleast_1d(np.loadtxt(model_flux_output_path, usecols=(1, ), dtype=float)))
                        
        if len(names_with_missing_rectified_model_flux) > 0:
            log.warn(f"The following {len(names_with_missing_rectified_model_flux)} are missing model fluxes: {names_with_missing_rectified_model_flux}")
        if len(names_with_missing_rectified_flux) > 0:
            log.warn(f"The following {len(names_with_missing_rectified_flux)} are missing rectified fluxes: {names_with_missing_rectified_flux}")

        #is_missing_model_flux = ~np.all(np.isfinite(model_flux), axis=1)

    else:
        is_missing_model_flux = np.zeros(N, dtype=bool)

    ferre_log_chi_sq = meta["log_chisq_fit"]
    ferre_log_snr_sq = meta["log_snr_sq"]
    
    is_missing_parameters = ~np.all(np.isfinite(parameters), axis=1)

    # Create some boolean flags. 
    header_path = control_kwds["SYNTHFILE(1)"]
    headers, *segment_headers = read_ferre_headers(expand_path(header_path))
    bad_lower = headers["LLIMITS"] + headers["STEPS"] / 8
    bad_upper = headers["ULIMITS"] - headers["STEPS"] / 8
    warn_lower = headers["LLIMITS"] + headers["STEPS"]
    warn_upper = headers["ULIMITS"] - headers["STEPS"]

    flag_grid_edge_bad = (parameters < bad_lower) | (parameters > bad_upper)
    flag_grid_edge_warn = (parameters < warn_lower) | (parameters > warn_upper)
    flag_ferre_fail = (parameters == -9999) | (e_parameters < -0.01) | ~np.isfinite(parameters)
    flag_any_ferre_fail = np.any(flag_ferre_fail, axis=1)
    flag_potential_ferre_timeout = is_missing_parameters
    flag_missing_model_flux = is_missing_model_flux | is_missing_rectified_model_flux

    # Get human-readable parameter names.
    to_human_readable_parameter_name = dict([(v, k) for k, v in TRANSLATE_LABELS.items()])
    parameter_names = [to_human_readable_parameter_name[k] for k in headers["LABEL"]]

    # TODO: we don't ahve any information about any continuum that was applied BEFORE ferre was executed.
    short_grid_name = parse_header_path(header_path)["short_grid_name"]

    common = dict(
        header_path=header_path, 
        short_grid_name=short_grid_name,
        pwd=dir, # TODO: Consider renaming
        ferre_n_obj=len(input_names),
        n_threads=control_kwds["NTHREADS"],
        interpolation_order=control_kwds["INTER"],
        continuum_reject=control_kwds.get("REJECTCONT", 0.0),
        continuum_order=control_kwds.get("NCONT", -1),
        continuum_flag=control_kwds.get("CONT", 0),
        continuum_observations_flag=control_kwds.get("OBSCONT", 0),
        f_format=control_kwds["F_FORMAT"],
        f_access=control_kwds["F_ACCESS"],
        weight_path=control_kwds["FILTERFILE"],
    )
    # Add frozen parameter flags.
    frozen_indices = set(range(1, 1 + len(parameter_names))).difference(set(map(int, control_kwds["INDV"].split())))
    for index in frozen_indices:
        common[f"flag_{parameter_names[index - 1]}_frozen"] = True

    ndim = int(control_kwds["NDIM"])
    for i, name in enumerate(input_names):
        name_meta = parse_ferre_spectrum_name(name)

        result = common.copy()
        result.update(
            source_pk=name_meta["source_pk"],
            spectrum_pk=name_meta["spectrum_pk"],
            initial_flags=name_meta["initial_flags"] or 0,
            #upstream_pk=name_meta["upstream_pk"],
            ferre_name=name,
            ferre_index=name_meta["index"],
            #ferre_output_index=i,
            rchi2=10**ferre_log_chi_sq[i], 
            penalized_rchi2=10**ferre_log_chi_sq[i],     
            ferre_log_snr_sq=ferre_log_snr_sq[i],
            flag_ferre_fail=flag_any_ferre_fail[i],
            flag_potential_ferre_timeout=flag_potential_ferre_timeout[i],
            flag_missing_model_flux=flag_missing_model_flux[i],
        )
        assert i == name_meta["index"]

        # Add correlation coefficients.
        #meta["cov"]
        # Add timing information, if we can.
        
        '''
        if not skip_pixel_arrays:
            snr = np.nanmedian(flux[i]/e_flux[i])
            result.update(
                snr=snr,
                flux=flux[i],
                e_flux=e_flux[i],
                model_flux=model_flux[i],
                rectified_flux=rectified_flux[i],
                rectified_model_flux=rectified_model_flux[i],
            )
        '''

        for j, parameter in enumerate(parameter_names):

            value = parameters[i, j]
            e_value = e_parameters[i, j]

            if e_value <= 0 or e_value >= 9999:
                e_value = np.nan
                flag_ferre_fail[i, j] = True # TODO: should we have more specific flags here?
            
            if parameter != "teff" and (value >= 9999 or value <= -9999):
                value = np.nan
                flag_ferre_fail[i, j] = True
                
            result.update({
                f"initial_{parameter}": input_parameters[i, j],
                parameter: value,
                f"e_{parameter}": e_value,
                f"flag_{parameter}_ferre_fail": flag_ferre_fail[i, j],
                f"flag_{parameter}_grid_edge_bad": flag_grid_edge_bad[i, j],
                f"flag_{parameter}_grid_edge_warn": flag_grid_edge_warn[i, j],
            })

        yield result
        