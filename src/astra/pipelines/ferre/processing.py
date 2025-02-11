import os
import numpy as np
from itertools import cycle
from typing import Iterable, Optional
from astra.pipelines.ferre.operator import post_execution_interpolation
from astra.pipelines.ferre import utils
from astra.utils import log, dict_to_list, expand_path

LARGE = 1e10 # TODO: This is also defined in pre_process, move it common

def _is_list_mode(path):
    return "input_list.nml" in path


# FERRE v4.8.8 src trunk : /uufs/chpc.utah.edu/common/home/sdss09/software/apogee/Linux/apogee/trunk/external/ferre/src

def pre_process_ferre(plans):
    processed = [_pre_process_ferre(**plan) for plan in plans]
    if len(plans) == 1:
        input_nml_path, total, n_threads, skipped = processed[0]
        pwd = os.path.dirname(input_nml_path)
        return (input_nml_path, pwd, total, n_threads, skipped)
    else:
        # Abundance mode.
        abundance_dir = os.path.dirname(os.path.dirname(processed[0][0]))

        input_nml_paths, total, skipped = ([], 0, [])
        for input_nml_path_, n_obj_, n_threads, skipped_ in processed:
            input_nml_paths.append(input_nml_path_[len(abundance_dir) + 1:]) # ppaths too long
            total += n_obj_
            for spectrum, kwds in skipped_:
                skipped.append((spectrum, kwds))        
        
        # Create a FERRE list file.
        input_nml_path = os.path.join(abundance_dir, "input_list.nml")
        with open(input_nml_path, "w") as fp:
            fp.write("\n".join(input_nml_paths) + "\n")

        pwd = os.path.dirname(input_nml_path)
        return (input_nml_path, pwd, total, n_threads, skipped)


def _pre_process_ferre(
    pwd: str,
    header_path: str,
    spectra,
    initial_teff: Iterable[float],
    initial_logg: Iterable[float],
    initial_m_h: Iterable[float],
    initial_log10_v_sini: Iterable[float] = None,
    initial_log10_v_micro: Iterable[float] = None,
    initial_alpha_m: Iterable[float] = None,
    initial_c_m: Iterable[float] = None,
    initial_n_m: Iterable[float] = None,
    initial_flags: Iterable[str] = None,
    upstream_pk: Iterable[int] = None,
    frozen_parameters: Optional[dict] = None,
    interpolation_order: int = 3,
    weight_path: Optional[str] = None,
    lsf_shape_path: Optional[str] = None,
    lsf_shape_flag: int = 0,
    error_algorithm_flag: int = 1,
    wavelength_interpolation_flag: int = 0,
    optimization_algorithm_flag: int = 3,
    pre_computed_continuum: Optional[Iterable[float]] = None,
    continuum_flag: int = 1,
    continuum_order: int = 4,
    continuum_segment: Optional[int] = None,
    continuum_reject: float = 0.3,
    continuum_observations_flag: int = 1,
    full_covariance: bool = True,
    pca_project: bool = False,
    pca_chi: bool = False,
    f_access: int = 0,
    f_format: int = 1,
    ferre_kwds: Optional[dict] = None,
    n_threads: int = 128,
    bad_pixel_flux_value: float = 1e-4,
    bad_pixel_error_value: float = 1e10,
    skyline_sigma_multiplier: float = 100,
    min_sigma_value: float = 0.05,
    spike_threshold_to_inflate_uncertainty: float = 3,
    reference_pixel_arrays_for_abundance_run=False,
    write_input_pixel_arrays=True,
    max_num_bad_pixels=2000,
    remove_existing_output_files=True,
    **kwargs
):

    if remove_existing_output_files:
        os.system(f"rm -f {pwd}/*.output* {pwd}/stdout* {pwd}/stderr*")    
    
    if kwargs:
        log.warning(f"astra.pipelines.ferre.pre_process.pre_process ignoring kwargs: {kwargs}")

    #n_threads = min(n_threads, len(spectra))

    # Validate the control file keywords.
    (
        control_kwds,
        headers,
        segment_headers,
        frozen_parameters,
    ) = utils.validate_ferre_control_keywords(
        header_path=header_path,
        frozen_parameters=frozen_parameters,
        interpolation_order=interpolation_order,
        weight_path=weight_path,
        lsf_shape_path=lsf_shape_path,
        lsf_shape_flag=lsf_shape_flag,
        error_algorithm_flag=error_algorithm_flag,
        wavelength_interpolation_flag=wavelength_interpolation_flag,
        optimization_algorithm_flag=optimization_algorithm_flag,
        continuum_flag=continuum_flag,
        continuum_order=continuum_order,
        continuum_segment=continuum_segment,
        continuum_reject=continuum_reject,
        continuum_observations_flag=continuum_observations_flag,
        full_covariance=full_covariance,
        pca_project=pca_project,
        pca_chi=pca_chi,
        n_threads=n_threads, # Limit threads to the number of objects
        f_access=f_access,
        f_format=f_format,
    )

    # Include any explicitly set ferre kwds
    control_kwds.update(ferre_kwds or dict())

    if reference_pixel_arrays_for_abundance_run:
        prefix = os.path.basename(pwd.rstrip("/")) + "/"
        for key in ("pfile", "opfile", "offile", "sffile"):
            control_kwds[key] = prefix + control_kwds[key]

    absolute_pwd = expand_path(pwd)

    values_or_cycle_none = lambda x: x if (x is not None and len(x) > 0) else cycle([None])
    all_initial_parameters = dict_to_list(dict(
        teff=values_or_cycle_none(initial_teff),
        logg=values_or_cycle_none(initial_logg),
        m_h=values_or_cycle_none(initial_m_h),
        log10_v_sini=values_or_cycle_none(initial_log10_v_sini),
        log10_v_micro=values_or_cycle_none(initial_log10_v_micro),
        alpha_m=values_or_cycle_none(initial_alpha_m),
        c_m=values_or_cycle_none(initial_c_m),
        n_m=values_or_cycle_none(initial_n_m),
        initial_flags=values_or_cycle_none(initial_flags),
        upstream_pk=values_or_cycle_none(upstream_pk)
    ))

    mask = utils.get_apogee_pixel_mask()

    has_warned_on_bad_pixels = False

    index, skipped, batch_names, batch_initial_parameters, batch_flux, batch_e_flux = (0, [], [], [], [], [])
    for (spectrum, initial_parameters) in zip(spectra, all_initial_parameters):

        if spectrum in skipped:
            continue

        if write_input_pixel_arrays:
            # We usually will be writing input pixel arrays, but sometimes we won't
            # (e.g., one other abundances execution has written the input pixel arrays
            # and this one could just be referencing them)

            # If this part fails, the spectrum doesn't exist and we should just continue
            try:
                with np.errstate(divide="ignore"):
                    flux = np.copy(spectrum.flux)
                    e_flux = np.copy(spectrum.ivar)**-0.5
            except:
                #log.warning(f"Exception accessing pixel arrays for spectrum {spectrum}")
                skipped.append((spectrum, {"flag_spectrum_io_error": True}))
                continue       
            

            try:
                pixel_flags = np.copy(spectrum.pixel_flags)
            except:
                warnings.warn(f"At least one spectrum has no pixel_flags attribute")
                pixel_flags = np.zeros(flux.shape, dtype=int)

            else:
                # TODO: move this to the ASPCAP coarse/stellar parameter section (before continuum norm).
                flux, e_flux = inflate_errors_at_bad_pixels(
                    flux,
                    e_flux,
                    pixel_flags,
                    skyline_sigma_multiplier=skyline_sigma_multiplier,
                    bad_pixel_flux_value=bad_pixel_flux_value,
                    bad_pixel_error_value=bad_pixel_error_value,
                    spike_threshold_to_inflate_uncertainty=spike_threshold_to_inflate_uncertainty,
                    min_sigma_value=min_sigma_value,
                )


            # Restrict by the number of bad pixels otherwise FERRE gets into trouble.
            if max_num_bad_pixels is not None:
                n_bad_pixels = np.sum((0 >= flux) | (e_flux >= 1e10))
                if n_bad_pixels >= max_num_bad_pixels:
                    if not has_warned_on_bad_pixels:
                        log.warning(f"Spectrum {spectrum} has too many bad pixels ({n_bad_pixels} > {max_num_bad_pixels}). Other spectra like this in the same FERRE job will have their warnings suppressed.")
                        has_warned_on_bad_pixels = True                        
                    skipped.append((spectrum, {"flag_too_many_bad_pixels": True}))
                    continue
                
            if pre_computed_continuum is not None:
                continuum = pre_computed_continuum[index]
                flux /= continuum
                e_flux /= continuum

            #bad = ((flux < 0) | (e_flux <= 0))
            #flux[bad] = 0.01
            e_flux = np.clip(e_flux, 0.005, np.inf)
            
            batch_flux.append(flux[mask])
            batch_e_flux.append(e_flux[mask])


        # make the initial flags 0 if None is given
        initial_flags = initial_parameters.pop("initial_flags") or 0
        upstream_pk = initial_parameters.pop("upstream_pk")

        batch_names.append(utils.get_ferre_spectrum_name(index, spectrum.source_pk, spectrum.spectrum_pk, initial_flags, upstream_pk))
        batch_initial_parameters.append(initial_parameters)
        index += 1


    batch_e_flux = np.array(batch_e_flux)
    if np.any(batch_e_flux < 0):
        bad = batch_e_flux < 0
        batch_e_flux[bad] = LARGE        
        log.warning(f"{np.sum(bad):.0} pixels had error values below 0!")
    
    #if len(skipped) > 0:
    #    log.warning(f"Skipping {len(skipped)} spectra ({100 * len(skipped) / len(spectra):.0f}%; of {len(spectra)})")

    if not batch_initial_parameters:
        return (pwd, 0, 0, skipped)


    synthfile_full_path = control_kwds["synthfile(1)"]
    if reference_pixel_arrays_for_abundance_run:
        control_kwds["synthfile(1)"] = os.path.basename(synthfile_full_path)
    else:
        control_kwds["synthfile(1)"] = os.path.basename(synthfile_full_path)

    control_kwds_formatted = utils.format_ferre_control_keywords(control_kwds, n_obj=1 + index)

    # Convert list of dicts of initial parameters to array.
    batch_initial_parameters_array = utils.validate_initial_and_frozen_parameters(
        headers,
        batch_initial_parameters,
        frozen_parameters,
        clip_initial_parameters_to_boundary_edges=True,
        clip_epsilon_percent=1,
    )
    # Create directory and write the control file        
    os.makedirs(absolute_pwd, exist_ok=True)
    target_path_prefix = None
    if reference_pixel_arrays_for_abundance_run:
        if write_input_pixel_arrays:
            target_path_prefix = f"{absolute_pwd}/../{os.path.basename(synthfile_full_path)}"[:-4]
    else:
        target_path_prefix = f"{absolute_pwd}/{os.path.basename(synthfile_full_path)}"[:-4]
    
    if target_path_prefix is not None:
        for suffix in ("hdr", "unf"):
            if not os.path.exists(f"{target_path_prefix}.{suffix}"):
                os.system(f"ln -s {synthfile_full_path[:-4]}.{suffix} {target_path_prefix}.{suffix}")


    with open(os.path.join(absolute_pwd, "input.nml"), "w") as fp:
        fp.write(control_kwds_formatted)       

    # hack: we do basename here in case we wrote the prefix to PFILE for the abundances run
    with open(os.path.join(absolute_pwd, os.path.basename(control_kwds["pfile"])), "w") as fp:
        for name, point in zip(batch_names, batch_initial_parameters_array):
            fp.write(utils.format_ferre_input_parameters(*point, name=name))

    if write_input_pixel_arrays:
        LARGE = 1e10

        batch_flux = np.array(batch_flux)
        batch_e_flux = np.array(batch_e_flux)

        if reference_pixel_arrays_for_abundance_run:
            flux_path = os.path.join(absolute_pwd, "../", control_kwds["ffile"])
            e_flux_path = os.path.join(absolute_pwd, "../", control_kwds["erfile"])
        else:
            flux_path = os.path.join(absolute_pwd, control_kwds["ffile"])
            e_flux_path = os.path.join(absolute_pwd, control_kwds["erfile"])

        non_finite_flux = ~np.isfinite(batch_flux)
        batch_flux[non_finite_flux] = 0.0
        batch_e_flux[non_finite_flux] = LARGE

        finite_e_flux = np.isfinite(batch_e_flux)
        batch_e_flux[~finite_e_flux] = LARGE
        if not np.any(finite_e_flux):
            log.warning(f"ALL flux errors are non-finite!")
            
        savetxt_kwds = dict(fmt="%.4e")#footer="\n")
        np.savetxt(flux_path, batch_flux, **savetxt_kwds)
        np.savetxt(e_flux_path, batch_e_flux, **savetxt_kwds)
    
    #if reference_pixel_arrays_for_abundance_run:
    #    for basename in (control_kwds["ffile"], control_kwds["erfile"]):
    #        os.system(f"ln -s {absolute_pwd}/../{basename} {absolute_pwd}/{basename}")
    n_obj = len(batch_names)
    return (f"{pwd}/input.nml", n_obj, min(n_threads, n_obj), skipped)


def inflate_errors_at_bad_pixels(
    flux,
    e_flux,
    bitfield,
    skyline_sigma_multiplier=100,
    bad_pixel_flux_value=1e-4,
    bad_pixel_error_value=1e10,
    spike_threshold_to_inflate_uncertainty=3,
    min_sigma_value=0.05,
):
    
    flux = np.copy(flux)
    e_flux = np.copy(e_flux)
    
    # Inflate errors around skylines,
    skyline_mask = (bitfield & 4096) > 0 # significant skyline
    e_flux[skyline_mask] *= skyline_sigma_multiplier

    # Sometimes FERRE will run forever.
    if spike_threshold_to_inflate_uncertainty > 0:

        flux_median = np.nanmedian(flux)
        flux_stddev = np.nanstd(flux)
        e_flux_median = np.median(e_flux)

        delta = (flux - flux_median) / flux_stddev
        is_spike = (delta > spike_threshold_to_inflate_uncertainty)
        #* (
        #    sigma_ < (parameters["spike_threshold_to_inflate_uncertainty"] * e_flux_median)
        #)
        #if np.any(is_spike):
        #    sum_spike = np.sum(is_spike)
            #fraction = sum_spike / is_spike.size
            #log.warning(
            #    f"Inflating uncertainties for {sum_spike} pixels ({100 * fraction:.2f}%) that were identified as spikes."
            #)
            #for pi in range(is_spike.shape[0]):
            #    n = np.sum(is_spike[pi])
            #    if n > 0:
            #        log.debug(f"  {n} pixels on spectrum index {pi}")
        e_flux[is_spike] = bad_pixel_error_value

    # Set bad pixels to have no useful data.
    if bad_pixel_flux_value is not None or bad_pixel_error_value is not None:                            
        bad = (
            ~np.isfinite(flux)
            | ~np.isfinite(e_flux)
            | (flux < 0)
            | (e_flux < 0)
            | ((bitfield & 16639) > 0) # any bad value (level = 1)
        )

        flux[bad] = bad_pixel_flux_value
        e_flux[bad] = bad_pixel_error_value        

    if min_sigma_value is not None:
        e_flux = np.clip(e_flux, min_sigma_value, np.inf)

    return (flux, e_flux)


def _get_ferre_chip_mask(observed_wavelength, chip_wavelengths):
    P = observed_wavelength.size
    mask = np.zeros(P, dtype=bool)
    for model_wavelength in chip_wavelengths:
        s_index = observed_wavelength.searchsorted(model_wavelength[0])
        e_index = s_index + model_wavelength.size
        mask[s_index:e_index] = True
    return mask                    


def post_process_ferre(input_nml_path, pwd, **kwargs) -> list[dict]:

    if _is_list_mode(input_nml_path):
        with open(input_nml_path, "r") as fp:
            input_nml_paths = [os.path.join(pwd, line.strip()) for line in fp.readlines()]
    else:
        input_nml_paths = [input_nml_path]

    # TODO: this might be slow we can probably use -l mode
    for path in input_nml_paths:
        post_execution_interpolation(path, pwd)
    
    v = []
    for path in input_nml_paths:
        v.extend(list(_post_process_ferre(path, pwd, skip_pixel_arrays=True, **kwargs)))
    return v


def _post_process_ferre(input_nml_path, pwd=None, skip_pixel_arrays=False, **kwargs) -> Iterable[dict]:
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

    pwd = pwd or os.path.dirname(input_nml_path)

    control_kwds = utils.read_control_file(input_nml_path)

    # Load input files.
    input_names, input_parameters = utils.read_input_parameter_file(pwd, control_kwds)   
    N = len(input_names)

    # Load and sort the rectified model flux path because this happens in abundances when we would normally use skip_pixel_arrays=True    
    parameter_input_path = os.path.join(pwd, control_kwds["PFILE"])
    for key in ("OFFILE", "OPFILE", "SFFILE"):
        path = os.path.join(pwd, control_kwds[key])
        if os.path.exists(path):
            os.system(f"vaffoff {parameter_input_path} {path}")

    parameters, e_parameters, meta, names_with_missing_outputs = utils.read_output_parameter_file(pwd, control_kwds, input_names)
    """
    except:
        # This only happens if the file does not exist
        D = int(control_kwds["NDIM"])
        parameters = np.nan * np.ones((N, D))
        e_parameters = np.ones_like(parameters)
        meta = {
            "log_snr_sq": np.nan * np.ones(N),
            "log_chisq_fit": np.nan * np.ones(N),
        }
        names_with_missing_outputs = input_names
    """

    # Create some boolean flags. 
    header_path = control_kwds["SYNTHFILE(1)"]
    if not os.path.exists(header_path):
        header_path = os.path.join(pwd, header_path)
    headers, *segment_headers = utils.read_ferre_headers(expand_path(header_path))
    bad_lower = headers["LLIMITS"] + headers["STEPS"] / 8
    bad_upper = headers["ULIMITS"] - headers["STEPS"] / 8
    warn_lower = headers["LLIMITS"] + headers["STEPS"]
    warn_upper = headers["ULIMITS"] - headers["STEPS"]

    flag_grid_edge_bad = (parameters < bad_lower) | (parameters > bad_upper)
    flag_grid_edge_warn = (parameters < warn_lower) | (parameters > warn_upper)
    flag_ferre_fail = (parameters == -9999) | (e_parameters < -0.01) | ~np.isfinite(parameters)
    flag_any_ferre_fail = np.any(flag_ferre_fail, axis=1)
    # TODO: handle these better:
    #flag_potential_ferre_timeout = is_missing_parameters
    #flag_missing_model_flux = is_missing_model_flux | is_missing_rectified_model_flux

    # Get human-readable parameter names.
    to_human_readable_parameter_name = dict([(v, k) for k, v in utils.TRANSLATE_LABELS.items()])
    parameter_names = [to_human_readable_parameter_name[k] for k in headers["LABEL"]]

    # TODO: we don't ahve any information about any continuum that was applied BEFORE ferre was executed.
    short_grid_name = utils.parse_header_path(header_path)["short_grid_name"]

    common = dict(
        header_path=header_path, 
        short_grid_name=short_grid_name,
        pwd=pwd, 
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
        name_meta = utils.parse_ferre_spectrum_name(name)

        result = common.copy()
        result.update(
            source_pk=name_meta["source_pk"],
            spectrum_pk=name_meta["spectrum_pk"],
            initial_flags=name_meta["initial_flags"] or 0,
            ferre_name=name,
            ferre_index=name_meta["index"],
            rchi2=10**meta["log_chisq_fit"][i], 
            penalized_rchi2=10**meta["log_chisq_fit"][i],     
            ferre_log_snr_sq=meta["log_snr_sq"][i],
            flag_ferre_fail=flag_any_ferre_fail[i],
            #flag_potential_ferre_timeout=flag_potential_ferre_timeout[i],
            #flag_missing_model_flux=flag_missing_model_flux[i],
        )
        assert i == name_meta["index"]

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
        

import numpy as np
import os
from collections import Counter

def get_suffix(input_nml_path):
    try:
        _, suffix = input_nml_path.split(".nml.")
    except ValueError:
        return 0
    else:
        return int(suffix)

def get_new_path(existing_path, new_suffix):
    if new_suffix == 1:
        return f"{existing_path}.{new_suffix}"
    else:
        return ".".join(existing_path.split(".")[:-1]) + f".{new_suffix}"


def re_process_partial_ferre(existing_input_nml_path, pwd=None, exclude_indices=None): 

    if pwd is None:
        pwd = os.path.dirname(existing_input_nml_path)
    
    existing_suffix = get_suffix(existing_input_nml_path)
    new_suffix = existing_suffix + 1

    new_input_nml_path = get_new_path(existing_input_nml_path, new_suffix)

    with open(existing_input_nml_path, "r") as f:
        lines = f.readlines()
    
    keys = ("PFILE", "OFFILE", "ERFILE", "OPFILE", "FFILE", "SFFILE")
    paths = {}
    for i, line in enumerate(lines):
        key = line.split("=")[0].strip()
        if key in keys:
            existing_relative_path = line.split("=")[1].strip("' \n")
            # All new relative paths must be within this directory
            # For example, if the flux arrays were stored in the parent directory, we must
            # store the new ones in THIS directory otherwise we could have two abundance directories
            # trying to write to the same parent file.
            # TODO: do that
            new_relative_path = get_new_path(existing_relative_path, new_suffix)
            lines[i] = line[:line.index("=")] + f"= '{new_relative_path}'"
            paths[key] = (existing_relative_path, new_relative_path)
    
    with open(new_input_nml_path, "w") as fp:
        fp.write("".join(lines))

    # TODO: copy input files to this directory because otherwise we will have partial flux files
    #       in the parent directory and it gets impossible to track
    
    # Find the things that are already written in all three output files.
    output_path_keys = ["OFFILE", "OPFILE"]
    if os.path.exists(os.path.join(pwd, paths["SFFILE"][0])):
        output_path_keys.append("SFFILE")

    counts = []
    for key in output_path_keys:
        names = np.unique(np.loadtxt(os.path.join(pwd, paths[key][0]), usecols=(0, ), dtype=str))
        counts.extend(names)

    completed_names = [k for k, v in Counter(counts).items() if v == len(output_path_keys)]
    input_names = np.loadtxt(os.path.join(pwd, paths["PFILE"][0]), usecols=(0, ), dtype=str)

    ignore_names = [] + completed_names
    if exclude_indices is not None:
        ignore_names.extend([input_names[int(idx)] for idx in exclude_indices])

    mask = [(name not in ignore_names) for name in input_names]
    if not any(mask):
        return (None, None)

    # Create new input files that ignore specific names.
    for key in ("PFILE", "ERFILE", "FFILE"):
        existing_path, new_path = paths[key]
        with open(os.path.join(pwd, existing_path), "r") as f:
            lines = f.readlines()
        
        with open(os.path.join(pwd, new_path), "w") as f:
            for line, m in zip(lines, mask):
                if m:
                    f.write(line)

    # Clean up the output files to only include things that are written in all three files.
    for key in output_path_keys:
        existing_path, new_path = paths[key]
        with open(os.path.join(pwd, existing_path), "r") as f:
            lines = f.readlines()
        
        lines = [line for line in lines if line.split()[0].strip() in completed_names]
        with open(os.path.join(pwd, existing_path) + ".cleaned", "w") as fp:
            fp.write("".join(lines))
        
    ignore_names = list(ignore_names)
    return (new_input_nml_path, ignore_names)
