"""Task for executing FERRE."""

import os
import numpy as np
import subprocess
import sys
import pickle
from itertools import cycle
from typing import Optional, Iterable
from tempfile import mkdtemp
from astropy.nddata import StdDevUncertainty
from collections import OrderedDict
from astra import __version__
from astra.base import task_decorator
from astra.tools.spectrum import Spectrum1D, SpectrumList
from astra.tools.spectrum.utils import spectrum_overlaps
from astra.contrib.ferre import bitmask, utils
from astra.utils import log, dict_to_list, list_to_dict, flatten, executable, expand_path, nested_list
from astra.database.astradb import (
    database,
    DataProduct,
    SDSSOutput
)
#from astra.sdss.datamodels.pipeline import create_pipeline_product
#from astra.sdss.datamodels.base import get_extname
from astra.contrib.ferre.bitmask import (PixelBitMask, ParamBitMask)
from peewee import BitField, FloatField, IntegerField, BooleanField

# FERRE v4.8.8 src trunk : /uufs/chpc.utah.edu/common/home/sdss09/software/apogee/Linux/apogee/trunk/external/ferre/src


 
"""
Any @task_decorator function that expects `Iterable[DataProduct]` for `data_product` can have other arguments
where `Iterable[...]` indicates one entry per `data_product`.
"""

def _prepare_ferre(
    pwd,
    header_path,
    data_product,
    hdu,
    initial_teff,
    initial_logg,
    initial_metals,
    initial_lgvsini,
    initial_log10vdop,
    initial_o_mg_si_s_ca_ti,
    initial_c,
    initial_n,
    initial_guess_source,
    frozen_parameters,
    interpolation_order,
    weight_path,
    lsf_shape_path,
    lsf_shape_flag,
    error_algorithm_flag,
    wavelength_interpolation_flag,
    optimization_algorithm_flag,
    continuum_flag,
    continuum_order,
    continuum_segment,
    continuum_reject,
    continuum_observations_flag,
    full_covariance,
    pca_project,
    pca_chi,
    f_access,
    f_format,
    ferre_kwds,
    n_threads,
    continuum_method,
    continuum_kwargs,
    bad_pixel_flux_value,
    bad_pixel_error_value,
    skyline_sigma_multiplier,
    min_sigma_value,
    spike_threshold_to_inflate_uncertainty,
    max_spectrum_per_data_product_hdu,
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

    # Write the control file
    with open(os.path.join(pwd, "input.nml"), "w") as fp:
        fp.write(control_kwds_formatted)       

    pixel_mask = PixelBitMask()

    # Construct mask to match FERRE model grid.
    chip_wavelengths = tuple(map(utils.wavelength_array, segment_headers))
    model_wavelengths = np.hstack(chip_wavelengths)


    # naming should be data_product_hdu_index

    # TODO: FLAGS to add:
    # https://github.com/sdss/apogee/blob/master/python/apogee/aspcap/ferre.py#L208
    # edge flas, warning, etc
    # fail flags
    '''
    class ParamBitMask(BitMask):
    #BitMask class for APOGEE ASPCAP bitmask (APOGEE_ASPCAPFLAG)

    flagname='APOGEE_PARAMFLAG'
    shorttitle='ParamBitMask'
    title='APOGEE_PARAMFLAG, APOGEE_ELEMFLAG : ASPCAP bitmask for individual parameters/abundances'
    blurb='These bitmasks are used to provide information indicate possible issues associated with individual measurements from <a href=”/dr17/irspec/aspcap/”>ASPCAP fits</a>. This bit provides more context for some of the bits in <code>ASPCAPFLAG</code>. A <code>PARAMFLAG</code> or <code>ELEMFLAG</code> is produced for each of the stellar parameters and chemical abundances measured in DR17, and each of the bitmasks has the same format. '
    name =['GRIDEDGE_BAD','CALRANGE_BAD','OTHER_BAD','FERRE_FAIL','PARAM_MISMATCH_BAD','FERRE_ERR_USED','TEFF_CUT','',
           'GRIDEDGE_WARN','CALRANGE_WARN','OTHER_WARN','FERRE_WARN','PARAM_MISMATCH_WARN','OPTICAL_WARN','ERR_WARN','FAINT_WARN',
           'PARAM_FIXED','RV_WARN','','','','','','',
           'SPEC_RC','SPEC_RGB','LOGG_CAL_MS','LOGG_CAL_RGB_MS','','','','RESERVED']
        '''

    # TODO: this part could be refactored to a function
    values_or_cycle_none = lambda x: x if x is not None else cycle([None])
    initial_parameters = dict_to_list(dict(
        teff=values_or_cycle_none(initial_teff),
        logg=values_or_cycle_none(initial_logg),
        metals=values_or_cycle_none(initial_metals),
        lgvsini=values_or_cycle_none(initial_lgvsini),
        log10vdop=values_or_cycle_none(initial_log10vdop),
        o_mg_si_s_ca_ti=values_or_cycle_none(initial_o_mg_si_s_ca_ti),
        c=values_or_cycle_none(initial_c),
        n=values_or_cycle_none(initial_n),
        initial_guess_source=values_or_cycle_none(initial_guess_source),
    ))

    batch_spectra, batch_name, batch_flux, batch_e_flux, batch_initial_parameters, n_obj = ({}, [], [], [], [], [])
    for i, (_data_product, _hdu, _initial_parameters) in enumerate(zip(data_product, hdu, initial_parameters)):

        # We only load spectra from the specific HDU given because `header_path` is a single file that includes the LSF model,
        # So everything with HDU=3 would be bundled together and everything with HDU=4 would be bundled together

        # Yes... because header_path is a single file that includes the LSF, so everything with HDU=3 would be bundled
        # together and everything with HDU=4 would be bundled together.

        spectra = SpectrumList.read(_data_product.path, hdu=_hdu)
        batch_spectra.setdefault((_data_product.id, _hdu), [])
        batch_spectra[(_data_product.id, _hdu)] = spectra

        for k, spectrum in enumerate(spectra):
            if max_spectrum_per_data_product_hdu is not None and max_spectrum_per_data_product_hdu >= k:
                break

            wavelength = spectrum.wavelength.value
            flux = spectrum.flux.value
            e_flux = spectrum.uncertainty.represent_as(StdDevUncertainty).array
            pixel_bitmask = spectrum.meta["BITMASK"]

            # Inflate errors around skylines, etc.
            skyline_mask = (pixel_bitmask & pixel_mask.get_value("SIG_SKYLINE")) > 0
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

            # NOTE: For IPL-2 we started inflating errors around skylines BEFORE doing continuum.
            #       In IPL-1 we inflated errors directly AFTER doing the continuum.

            # Perform any continuum rectification pre-processing.
            if continuum_method is not None:
                kwds = continuum_kwargs.copy()
                if continuum_method == "astra.contrib.aspcap.continuum.MedianFilter":
                    kwds.update(upstream_task_id=_initial_parameters["initial_guess_source"])
                f_continuum = executable(continuum_method)(**kwds)

                f_continuum.fit(spectrum)
                continuum = f_continuum(spectrum)
                flux /= continuum
                e_flux /= continuum
            else:
                f_continuum = None     
            
            # Set bad pixels to have no useful data.
            if bad_pixel_flux_value is not None or bad_pixel_error_value is not None:                            
                bad = (
                    ~np.isfinite(flux)
                    | ~np.isfinite(e_flux)
                    | (flux < 0)
                    | (e_flux < 0)
                    | ((pixel_bitmask & pixel_mask.get_level_value(1)) > 0)
                )

                flux[bad] = bad_pixel_flux_value
                e_flux[bad] = bad_pixel_error_value

            # Clip the error array. This is a pretty bad idea but I am doing what was done before!
            if min_sigma_value is not None:
                e_flux = np.clip(e_flux, min_sigma_value, np.inf)

            # Retrict to the pixels within the model wavelength grid.
            mask = _get_ferre_chip_mask(wavelength, chip_wavelengths)

            batch_flux.append(flux[mask])
            batch_e_flux.append(e_flux[mask])
            # Repeat the initial parameters for the N spectra in this data product
            batch_initial_parameters.append(_initial_parameters)

            batch_name.append(f"{i}_{k}_{_data_product.id}_{_hdu}")
        
        n_obj.append(1 + k)

    # Convert list of dicts of initial parameters to array.
    batch_initial_parameters_array = utils.validate_initial_and_frozen_parameters(
        headers,
        batch_initial_parameters,
        frozen_parameters,
        clip_initial_parameters_to_boundary_edges=True,
        clip_epsilon_percent=1,
    )

    with open(os.path.join(pwd, control_kwds["pfile"]), "w") as fp:
        for name, point in zip(batch_name, batch_initial_parameters_array):
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
    return n_obj



def _post_process_ferre(pwd, model, data_product=None, ferre_timeout=None):
    
    stdout_path, stderr_path = _stdout_stderr_paths(pwd)

    with open(stdout_path, "r") as fp:
        stdout = fp.read()
    with open(stderr_path, "r") as fp:
        stderr = fp.read()

    n_done, n_error, control_kwds, meta = utils.parse_ferre_output(pwd, stdout, stderr)

    log.info(f"Found {n_done} completed successfully and {n_error} errors")
    
    # Parse outputs.
    path = os.path.join(pwd, control_kwds["PFILE"])
    input_names = np.atleast_1d(np.loadtxt(path, usecols=(0, ), dtype=str))

    # FFILE and ERFILE are inputs, so they will always be the right shape.
    try:
        path = os.path.join(pwd, control_kwds["FFILE"])
        flux = np.atleast_2d(np.loadtxt(path))
    except:
        log.exception(f"Failed to load input flux from {path}")
    try:
        path = os.path.join(pwd, control_kwds["ERFILE"])
        flux_sigma = np.atleast_2d(np.loadtxt(path))
    except:
        log.exception(f"Failed to load flux sigma from {path}")
        raise

    # Now parse the outputs from the FERRE run.
    path = os.path.join(pwd, control_kwds["OPFILE"])
    try:
        output_names, output_params, output_param_errs, meta = utils.read_output_parameter_file(
            path,
            n_dimensions=control_kwds["NDIM"],
            full_covariance=control_kwds["COVPRINT"],
        )
    except:
        log.exception(f"Exception when parsing FERRE output parameter file {path}")
        raise

    if len(input_names) > len(output_names):
        log.warning(f"Number of input parameters does not match output parameters ({len(input_names)} > {len(output_names)}). FERRE may have failed. We will pick up the pieces..")

    # Which entries are missing?
    missing_names = list(set(input_names).difference(output_names))
    missing_indices = [np.where(input_names == mn)[0][0] for mn in missing_names]
    for i in np.argsort(missing_indices):
        missing_name, missing_index = (missing_names[i], missing_indices[i])
        log.warning(f"Missing parameters for spectrum named {missing_name} (index {missing_index}; row {missing_index+1})")        

    # We will fill the missing parameters with nans, and missing fluxes with nans too
    N, P = flux.shape
    D = int(control_kwds["NDIM"]) 
    params = np.nan * np.ones((N, D), dtype=float)
    param_errs = np.nan * np.ones((N, D), dtype=float)
    log_chisq_fit = np.nan * np.ones(N)
    log_snr_sq = np.nan * np.ones(N)
    frac_phot_data_points = np.nan * np.ones(N)

    indices = []
    for i, name in enumerate(output_names):
        index, = np.where(input_names == name)
        assert len(index) == 1, f"Name {name} (index {i}) appears more than once in the input parameter file!"
        indices.append(index[0])
    indices = np.array(indices)

    params[indices] = output_params
    param_errs[indices] = output_param_errs
    log_chisq_fit[indices] = meta["log_chisq_fit"]
    log_snr_sq[indices] = meta["log_snr_sq"]
    frac_phot_data_points[indices] = meta["frac_phot_data_points"]


    offile_path = os.path.join(pwd, control_kwds["OFFILE"])
    _model_flux = np.atleast_2d(np.loadtxt(offile_path, usecols=1 + np.arange(P)))
    _model_flux_names = np.atleast_1d(np.loadtxt(offile_path, usecols=(0, ), dtype=str))
    model_indices = []
    for i, name in enumerate(_model_flux_names):
        index, = np.where(input_names == name)
        model_indices.append(index[0])
    model_indices = np.array(model_indices)

    model_flux = np.nan * np.ones((N, P), dtype=float)
    model_flux[model_indices] = _model_flux

    if "SFFILE" in control_kwds:
        try:
            sffile_path = os.path.join(pwd, control_kwds["SFFILE"])
            _normalized_flux = np.atleast_2d(np.loadtxt(sffile_path, usecols=1 + np.arange(P)))
            _normalized_flux_names = np.atleast_1d(np.loadtxt(sffile_path, usecols=(0, ), dtype=str))
        except:
            log.exception(f"Failed to load normalized observed flux from {sffile_path}")
            raise
        else:
            # Order the normalized flux to be the same as the inputs
            normalized_flux_indices = []
            for i, name in enumerate(_normalized_flux_names):
                index, = np.where(input_names == name)
                normalized_flux_indices.append(index[0])
            normalized_flux_indices = np.array(normalized_flux_indices)

            normalized_flux = np.nan * np.ones((N, P), dtype=float)
            normalized_flux[normalized_flux_indices] = _normalized_flux

            continuum = flux / normalized_flux
    else:
        continuum = np.ones_like(flux)
        normalized_flux = flux

    has_complete_results = (
        np.any(np.isfinite(params), axis=1)
    *   np.any(np.isfinite(model_flux), axis=1)
    )

    fill_value = -999
    # If we only have some things (eg params but no model flux) we should make it all nan,
    # ebcause we dont want to rely on this downstream
    params[~has_complete_results] = fill_value
    model_flux[~has_complete_results] = fill_value
    normalized_flux[~has_complete_results] = fill_value
    continuum[~has_complete_results] = fill_value
    
    header_path = control_kwds["SYNTHFILE(1)"]
    headers, *segment_headers = utils.read_ferre_headers(
        utils.expand_path(header_path)
    )
    parameter_names = utils.sanitise(headers["LABEL"])

    # Flag things.
    param_bitmask = bitmask.ParamBitMask()
    param_bitmask_flags = np.zeros(params.shape, dtype=np.int64)

    bad_lower = headers["LLIMITS"] + headers["STEPS"] / 8
    bad_upper = headers["ULIMITS"] - headers["STEPS"] / 8
    param_bitmask_flags[
        (params < bad_lower) | (params > bad_upper)
    ] |= param_bitmask.get_value("GRIDEDGE_BAD")

    warn_lower = headers["LLIMITS"] + headers["STEPS"]
    warn_upper = headers["ULIMITS"] - headers["STEPS"]
    param_bitmask_flags[
        (params < warn_lower) | (params > warn_upper)
    ] |= param_bitmask.get_value("GRIDEDGE_WARN")
    param_bitmask_flags[
        (params == fill_value) | (param_errs < -0.01) | ~np.isfinite(params)
    ] |= param_bitmask.get_value("FERRE_FAIL")
    
    # Check for any erroneous outputs
    if np.any(param_bitmask_flags & param_bitmask.get_value("FERRE_FAIL")):
        v = param_bitmask_flags & param_bitmask.get_value("FERRE_FAIL")
        idx = np.where(
            np.any(
                param_bitmask_flags & param_bitmask.get_value("FERRE_FAIL"), axis=1
            )
        )
        log.warning(f"FERRE returned all erroneous values for an entry: {idx} {v}")

    # We need to yield N times, where N is len(data_products) == len(hdu), etc.
    # So we need to link things by (i, k) where i is the index of the data product
    # and k is the index of the segment.
    si = 0
    ferre_n_obj = params.shape[0]
    
    try:
        timings = utils.get_processing_times(stdout)
        ferre_times_elapsed = timings["time_per_spectrum"][indices]
        ferre_time_load = timings["time_load"]
    except:
        ferre_time_load = None
        ferre_times_elapsed = cycle([None])

    # Get data products if necessary.
    log.info(f"Collecting data products")
    if data_product is None:
        data_product_ids = list(set(map(int, (ea.split("_")[2] for ea in input_names))))
        dps = (
            DataProduct
            .select()
            .where(DataProduct.id << data_product_ids)
        )
        data_product_dict = {}
        for data_product in dps:
            data_product_dict[data_product.id] = data_product
    else:
        data_product_dict = {}
        for data_product in data_product:
            data_product_dict[data_product.id] = data_product

    # Load spectra.
    log.info(f"Loading spectra")
    spectra = {}
    for input_name in input_names:
        z, k, data_product_id, hdu = map(int, input_name.split("_"))

        key = (data_product_id, hdu)
        if key not in spectra: 
            spectra[key] = SpectrumList.read(data_product_dict[data_product_id].path, hdu=hdu)
    
    log.info("Yield results")

    # outputs must be per data_product_id, hdu
    # TODO: This has a strong implicit assumption of 1 spectrum per spectrumlist.
    #       wwill need to track number of objects per data_product and hdu so that we
    #       yield back the right things in the right order.
    for z, k, data_product_id, hdu in map(lambda _: map(int, _.split("_")), input_names):

        key = (data_product_id, hdu)
        spectrum = spectra[key][k]

        # TODO: check for failure.
        result = dict(zip(parameter_names, params[z]))
        result.update(dict(zip([f"e_{pn}" for pn in parameter_names], param_errs[z])))
        result.update(dict(zip([f"bitmask_{pn}" for pn in parameter_names], param_bitmask_flags[z])))
        result.update(
            log_chisq_fit=log_chisq_fit[z],
            log_snr_sq=log_snr_sq[z], 
            frac_phot_data_points=frac_phot_data_points[z],
        )
        try:
            ferre_time_elapsed = ferre_times_elapsed[z]
        except:
            ferre_time_elapsed = None
        
        output = model(
            data_product=data_product_dict[data_product_id],
            spectrum=spectrum,
            ferre_n_obj=ferre_n_obj,
            ferre_time_load=ferre_time_load,
            ferre_time_elapsed=ferre_time_elapsed,
            ferre_timeout=ferre_timeout,
            **result
        )

        output_path = os.path.join(pwd, f"{output.task.id}.pkl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as fp:
            pickle.dump((model_flux[z], continuum[z]), fp, -1)
        
        output_data_product, _ = DataProduct.get_or_create(
            release="sdss5",
            filetype="full",
            kwargs=dict(full=output_path)
        )
        output.output_data_product = output_data_product        

        print(f"yielding {key} and {output} with {type(output.data_product)}: {output.data_product}")
        yield output
    
    #for key, output in outputs.items():
    #    print(f"yielding {key} and {output}")
    #    yield output


def ferre(
    model,
    pwd: str,
    header_path: str,
    data_product: Iterable[DataProduct],
    hdu: Iterable[int],
    initial_teff: Iterable[float],
    initial_logg: Iterable[float],
    initial_metals: Iterable[float],
    initial_lgvsini: Iterable[float] = None,
    initial_log10vdop: Iterable[float] = None,
    initial_o_mg_si_s_ca_ti: Iterable[float] = None,
    initial_c: Iterable[float] = None,
    initial_n: Iterable[float] = None,
    initial_guess_source: Iterable[str] = None,
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
    max_spectrum_per_data_product_hdu: Optional[int] = None,
):

    assert pwd is not None
    '''
    if pwd is None:

        os.makedirs(DEFAULT_FERRE_pwd, exist_ok=True)
        pwd = mkdtemp(dir=DEFAULT_FERRE_pwd)
        log.warning(f"Setting random parent directory for FERRE run: {pwd}")
    '''
        
    pwd = expand_path(pwd)
    os.makedirs(pwd, exist_ok=True)

    pwd 
    log.info(f"FERRE working directory: {pwd}")

    n_obj = _prepare_ferre(
        pwd,
        header_path=header_path,
        data_product=data_product,
        hdu=hdu,    
        initial_teff=initial_teff,
        initial_logg=initial_logg,
        initial_metals=initial_metals,
        initial_lgvsini=initial_lgvsini,
        initial_log10vdop=initial_log10vdop,
        initial_o_mg_si_s_ca_ti=initial_o_mg_si_s_ca_ti,
        initial_c=initial_c,
        initial_n=initial_n,
        initial_guess_source=initial_guess_source,
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
        f_access=f_access,
        f_format=f_format,
        ferre_kwds=ferre_kwds,
        n_threads=n_threads,
        continuum_method=continuum_method,
        continuum_kwargs=continuum_kwargs,
        bad_pixel_flux_value=bad_pixel_flux_value,
        bad_pixel_error_value=bad_pixel_error_value,
        skyline_sigma_multiplier=skyline_sigma_multiplier,
        min_sigma_value=min_sigma_value,
        spike_threshold_to_inflate_uncertainty=spike_threshold_to_inflate_uncertainty,
        max_spectrum_per_data_product_hdu=max_spectrum_per_data_product_hdu,
    )

    est_load_time = 60 # if using n_threads = 128
    est_wall_time = max(1_800, 1_200 * max(1, n_obj / n_threads))

    timeout = min(23 * 60 * 60, est_wall_time + est_load_time)

    ferre_timeout = False
    stdout_path, stderr_path = _stdout_stderr_paths(pwd)
    try:
        with open(stdout_path, "w") as stdout:
            with open(stderr_path, "w") as stderr:
                process = subprocess.run(
                    ["ferre.x"],
                    cwd=pwd,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                    timeout=timeout, 
                )
    except subprocess.TimeoutExpired:
        log.exception(f"FERRE has timed out in {pwd}")
        ferre_timeout = True
        log.info(f"We will try to collect the results that we can")
    
    except:
        log.exception(f"Exception when calling FERRE in {pwd}:")
        log.info(f"Will continue to try and recover what we can")
        raise

    finally:
        yield from _post_process_ferre(pwd, model, ferre_timeout=ferre_timeout)


def _stdout_stderr_paths(pwd):
    stdout_path = os.path.join(pwd, "stdout")
    stderr_path = os.path.join(pwd, "stderr")
    return (stdout_path, stderr_path)

def _get_ferre_chip_mask(observed_wavelength, chip_wavelengths):
    P = observed_wavelength.size
    mask = np.zeros(P, dtype=bool)
    for model_wavelength in chip_wavelengths:
        s_index = observed_wavelength.searchsorted(model_wavelength[0])
        e_index = s_index + model_wavelength.size
        mask[s_index:e_index] = True
    return mask                    



if __name__ == "__main__":

    from astra.database.astradb import DataProduct

    dp = DataProduct.get(filetype="apStar")

    header_path = "$SAS_BASE_DIR/dr17/apogee/spectro/speclib/synth/synspec/marcs/giantisotopes/sgGK_200921nlte_lsfa/p_apssgGK_200921nlte_lsfa_012_075.hdr"


    foo = list(ferre(
        [dp], 
        header_path=header_path,
        initial_parameters=[
            {"TEFF": 5777, "LOGG": 4.4, "METALS": 0.0}
        ])
    )
