import os
import numpy as np
from typing import Iterable

from astra.utils import log, expand_path
from astra.pipelines.ferre.utils import (
    read_ferre_headers,
    read_control_file, 
    read_input_parameter_file,
    read_output_parameter_file,
    read_and_sort_output_data_file,
    parse_ferre_spectrum_name,
    parse_header_path,
    TRANSLATE_LABELS
)


def post_process_ferre(pwd) -> Iterable[dict]:
    """
    Post-process results from a FERRE execution.

    :param pwd:
        The working directory of the FERRE execution.
    """

    pwd = expand_path(pwd)
    # TODO: Put this somewhere common?
    stdout_path = os.path.join(pwd, "stdout")

    if os.path.exists(stdout_path):
        with open(stdout_path, "r") as fp:
            stdout = fp.read()
        # Parse timing
    else:
        log.warning(f"No stdout file found at {stdout_path}. No timing information available.")
        timing = {}

    control_kwds = read_control_file(os.path.join(pwd, "input.nml"))

    # Load input files.
    input_names, input_parameters = read_input_parameter_file(pwd, control_kwds)   
    flux = np.atleast_2d(np.loadtxt(os.path.join(pwd, control_kwds["FFILE"])))
    e_flux = np.atleast_2d(np.loadtxt(os.path.join(pwd, control_kwds["ERFILE"])))

    model_flux, names_with_missing_model_flux, output_model_flux_indices = read_and_sort_output_data_file(
        os.path.join(pwd, control_kwds["OFFILE"]), 
        input_names
    )
    rectified_flux, names_with_missing_rectified_flux, output_rectified_model_flux_indices = read_and_sort_output_data_file(
        os.path.join(pwd, control_kwds["SFFILE"]),
        input_names
    )
    assert np.all(output_model_flux_indices == output_rectified_model_flux_indices)
    parameters, e_parameters, meta, names_with_missing_outputs = read_output_parameter_file(pwd, control_kwds, input_names)

    if names_with_missing_model_flux:
        log.warn(f"The following {len(names_with_missing_model_flux)} are missing model fluxes: {names_with_missing_model_flux}")
    if names_with_missing_rectified_flux:
        log.warn(f"The following {len(names_with_missing_rectified_flux)} are missing rectified fluxes: {names_with_missing_rectified_flux}")
    if names_with_missing_outputs:
        log.warn(f"The following {len(names_with_missing_outputs)} are missing outputs: {names_with_missing_outputs}")
    
    log_chisq_fit = meta["log_chisq_fit"]
    log_snr_sq = meta["log_snr_sq"]
    frac_phot_data_points = meta["frac_phot_data_points"]
    
    
    is_missing_parameters = ~np.all(np.isfinite(parameters), axis=1)
    is_missing_model_flux = ~np.all(np.isfinite(model_flux), axis=1)
    is_missing_rectified_flux = ~np.all(np.isfinite(rectified_flux), axis=1)

    # Create some boolean flags. 
    header_path = control_kwds["SYNTHFILE(1)"]
    headers, *segment_headers = read_ferre_headers(expand_path(header_path))
    bad_lower = headers["LLIMITS"] + headers["STEPS"] / 8
    bad_upper = headers["ULIMITS"] - headers["STEPS"] / 8
    warn_lower = headers["LLIMITS"] + headers["STEPS"]
    warn_upper = headers["ULIMITS"] - headers["STEPS"]

    flag_grid_edge_bad = (parameters < bad_lower) | (parameters > bad_upper)
    flag_grid_edge_warn = (parameters < warn_lower) | (parameters > warn_upper)
    flag_ferre_fail = (parameters == -9999) | (e_parameters < -0.01)
    flag_any_ferre_fail = np.any(flag_ferre_fail, axis=1)
    flag_potential_ferre_timeout = is_missing_parameters
    flag_missing_model_flux = is_missing_model_flux | is_missing_rectified_flux

    # Get human-readable parameter names.
    to_human_readable_parameter_name = dict([(v, k) for k, v in TRANSLATE_LABELS.items()])
    parameter_names = [to_human_readable_parameter_name[k] for k in headers["LABEL"]]

    # TODO: we don't ahve any information about any continuum that was applied BEFORE ferre was executed.
    short_grid_name = parse_header_path(header_path)["short_grid_name"]

    common = dict(
        header_path=header_path, 
        short_grid_name=short_grid_name,
        pwd=pwd,
        ferre_n_obj=len(input_names),
        n_threads=control_kwds["NTHREADS"],
        interpolation_order=control_kwds["INTER"],
        continuum_reject=control_kwds["REJECTCONT"],
        continuum_order=control_kwds["NCONT"],
        f_format=control_kwds["F_FORMAT"],
        f_access=control_kwds["F_ACCESS"],
        weight_path=control_kwds["FILTERFILE"],
    )
    # Add frozen parameter flags.
    frozen_indices = set(range(1, 1 + len(parameter_names))).difference(set(map(int, control_kwds["INDV"].split())))
    for index in frozen_indices:
        common[f"flag_{parameter_names[index - 1]}_frozen"] = True

    for i, name in enumerate(input_names):
        name_meta = parse_ferre_spectrum_name(name)

        result = common.copy()
        result.update(
            sdss_id=name_meta["sdss_id"],
            spectrum_id=name_meta["spectrum_id"],
            initial_flags=name_meta["initial_flags"],
            upstream_id=name_meta["upstream_id"],
            ferre_name=name,
            ferre_input_index=name_meta["index"],
            ferre_output_index=output_model_flux_indices[i],
            ferre_log_chisq=log_chisq_fit[i], 
            ferre_log_snr_sq=log_snr_sq[i],
            ferre_log_penalized_chisq=log_chisq_fit[i],     
            frac_phot_data_points=frac_phot_data_points[i],      
            flag_ferre_fail=flag_any_ferre_fail[i],
            flag_potential_ferre_timeout=flag_potential_ferre_timeout[i],
            flag_missing_model_flux=flag_missing_model_flux[i],
        )

        result.update(
            flux=flux[i],
            e_flux=e_flux[i],
            model_flux=model_flux[i],
            rectified_flux=rectified_flux[i],
        )

        for j, parameter in enumerate(parameter_names):
            result.update({
                f"initial_{parameter}": input_parameters[i, j],
                parameter: parameters[i, j],
                f"e_{parameter}": e_parameters[i, j],
                f"flag_{parameter}_ferre_fail": flag_ferre_fail[i, j],
                f"flag_{parameter}_grid_edge_bad": flag_grid_edge_bad[i, j],
                f"flag_{parameter}_grid_edge_warn": flag_grid_edge_warn[i, j],
            })

        # TODO: Load metadata from pwd/meta.json (e.g., pre-continuum steps)
        # TODO: Include correlation coefficients?
        # TODO: Include timing from ferre
        yield result
        