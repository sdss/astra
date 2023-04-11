"""Task for executing FERRE."""

import os
import numpy as np
from itertools import cycle
from typing import Optional, Iterable
from astra.pipelines.ferre import utils
from astra.models import Spectrum
from astra.utils import log, dict_to_list, expand_path

# FERRE v4.8.8 src trunk : /uufs/chpc.utah.edu/common/home/sdss09/software/apogee/Linux/apogee/trunk/external/ferre/src

def pre_process_ferre(
    pwd: str,
    header_path: str,
    spectra: Iterable[Spectrum],
    initial_teff: Iterable[float],
    initial_logg: Iterable[float],
    initial_m_h: Iterable[float],
    initial_log10_v_sini: Iterable[float] = None,
    initial_log10_v_micro: Iterable[float] = None,
    initial_alpha_m: Iterable[float] = None,
    initial_c_m: Iterable[float] = None,
    initial_n_m: Iterable[float] = None,
    initial_flags: Iterable[str] = None,
    frozen_parameters: Optional[dict] = None,
    interpolation_order: int = 3,
    weight_path: Optional[str] = None,
    lsf_shape_path: Optional[str] = None,
    lsf_shape_flag: int = 0,
    error_algorithm_flag: int = 1,
    wavelength_interpolation_flag: int = 0,
    optimization_algorithm_flag: int = 3,
    continuum_flag: int = 1,
    continuum_order: int = 4,
    continuum_segment: Optional[int] = None,
    continuum_reject: float = 0.3,
    continuum_observations_flag: int = 1,
    full_covariance: bool = False,
    pca_project: bool = False,
    pca_chi: bool = False,
    f_access: int = 0,
    f_format: int = 1,
    ferre_kwds: Optional[dict] = None,
    n_threads: int = 1,
    continuum_method: Optional[str] = None,
    continuum_kwargs: Optional[dict] = None,
    bad_pixel_flux_value: float = 1e-4,
    bad_pixel_error_value: float = 1e10,
    skyline_sigma_multiplier: float = 100,
    min_sigma_value: float = 0.05,
    spike_threshold_to_inflate_uncertainty: float = 3,
):

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
        n_threads=n_threads,
        f_access=f_access,
        f_format=f_format,
    )

    # Include any explicitly set ferre kwds
    control_kwds.update(ferre_kwds or dict())
    control_kwds_formatted = utils.format_ferre_control_keywords(control_kwds)
    log.info(f"FERRE control keywords:\n{control_kwds_formatted}")

    pwd = expand_path(pwd)
    os.makedirs(pwd, exist_ok=True)
    log.info(f"FERRE working directory: {pwd}")

    # Write the control file
    with open(os.path.join(pwd, "input.nml"), "w") as fp:
        fp.write(control_kwds_formatted)       

    # Construct mask to match FERRE model grid.
    chip_wavelengths = tuple(map(utils.wavelength_array, segment_headers))
    
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
    ))

    # Retrict to the pixels within the model wavelength grid.
    # TODO: Assuming all spectra are the same.
    mask = _get_ferre_chip_mask(spectra[0].wavelength, chip_wavelengths)

    batch_names, batch_initial_parameters, batch_flux, batch_e_flux = ([], [], [], [])
    for i, (spectrum, initial_parameters) in enumerate(zip(spectra, all_initial_parameters)):

        flux = np.copy(spectrum.flux)
        e_flux = np.copy(spectrum.ivar)**-0.5
        try:
            flag_pixels = spectrum.flag_pixels
        except AttributeError:
            log.warn(f"Spectrum {spectrum} has no flag_pixels attribute")

        else:                
            inflate_errors_at_bad_pixels(
                flux,
                e_flux,
                flag_pixels,
                skyline_sigma_multiplier=skyline_sigma_multiplier,
                bad_pixel_flux_value=bad_pixel_flux_value,
                bad_pixel_error_value=bad_pixel_error_value,
                spike_threshold_to_inflate_uncertainty=spike_threshold_to_inflate_uncertainty,
                min_sigma_value=min_sigma_value,
            )

        # Perform any continuum rectification pre-processing.
        if continuum_method is not None:
            kwds = continuum_kwargs.copy()
            if continuum_method == "astra.contrib.aspcap.continuum.MedianFilter":
                kwds.update(upstream_task_id=_initial_parameters["initial_flags"])
            f_continuum = executable(continuum_method)(**kwds)

            f_continuum.fit(spectrum)
            continuum = f_continuum(spectrum)
            flux /= continuum
            e_flux /= continuum
        else:
            f_continuum = None     
        
        batch_flux.append(flux[mask])
        batch_e_flux.append(e_flux[mask])

        initial_flags = initial_parameters.pop("initial_flags")
        batch_names.append(utils.get_ferre_spectrum_name(i, spectrum.source_id, spectrum.spectrum_id, initial_flags))
        batch_initial_parameters.append(initial_parameters)
    

    # Convert list of dicts of initial parameters to array.
    batch_initial_parameters_array = utils.validate_initial_and_frozen_parameters(
        headers,
        batch_initial_parameters,
        frozen_parameters,
        clip_initial_parameters_to_boundary_edges=True,
        clip_epsilon_percent=1,
    )

    with open(os.path.join(pwd, control_kwds["pfile"]), "w") as fp:
        for name, point in zip(batch_names, batch_initial_parameters_array):
            fp.write(utils.format_ferre_input_parameters(*point, name=name))

    batch_flux = np.array(batch_flux)
    batch_e_flux = np.array(batch_e_flux)

    # Write data arrays.
    savetxt_kwds = dict(fmt="%.4e", footer="\n")
    np.savetxt(
        os.path.join(pwd, control_kwds["ffile"]), batch_flux, **savetxt_kwds
    )
    np.savetxt(
        os.path.join(pwd, control_kwds["erfile"]), batch_e_flux, **savetxt_kwds
    )

    n_obj = batch_flux.shape[0]
    return (pwd, n_obj)



def inflate_errors_at_bad_pixels(
    flux,
    e_flux,
    bitfield,
    skyline_sigma_multiplier,
    bad_pixel_flux_value,
    bad_pixel_error_value,
    spike_threshold_to_inflate_uncertainty,
    min_sigma_value,
):

    # Inflate errors around skylines,
    skyline_mask = (bitfield & pixel_mask.get_value("SIG_SKYLINE")) > 0
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
            | ((bitfield & pixel_mask.get_level_value(1)) > 0)
        )

        flux[bad] = bad_pixel_flux_value
        e_flux[bad] = bad_pixel_error_value        

    if min_sigma_value is not None:
        e_flux = np.clip(e_flux, min_sigma_value, np.inf)

    return None


def _get_ferre_chip_mask(observed_wavelength, chip_wavelengths):
    P = observed_wavelength.size
    mask = np.zeros(P, dtype=bool)
    for model_wavelength in chip_wavelengths:
        s_index = observed_wavelength.searchsorted(model_wavelength[0])
        e_index = s_index + model_wavelength.size
        mask[s_index:e_index] = True
    return mask                    