import numpy as np
import multiprocessing as mp
import os
import re
import subprocess
import sys
import threading

from collections import OrderedDict
from io import StringIO, BytesIO
from inspect import getfullargspec
from shutil import rmtree
from tempfile import mkdtemp
from time import sleep, time
from tqdm import tqdm

from astra.utils import log
from astra.contrib.ferre import utils, bitmask

# Cross-check
# /uufs/chpc.utah.edu/common/home/sdss50/dr17/apogee/spectro/aspcap/dr17/synspec/bundle_apo25m/apo25m_003/ferre/elem_K

import json

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def prepare_ferre(directory, input_kwds):
    json_kwds = dict(indent=2, cls=NumpyEncoder)
    log.debug(f"Parameters supplied to FERRE:")
    log.debug(json.dumps((input_kwds["initial_parameters"], input_kwds["frozen_parameters"]), **json_kwds))

    # Parse and validate parameters.
    wavelength, flux, sigma, mask, names, initial_parameters, kwds, meta = parsed_kwds = utils.parse_ferre_inputs(**input_kwds)

    log.debug(f"Parameters after parsing FERRE:")
    log.debug(f"Initial parameters: {json.dumps(initial_parameters, **json_kwds)}")
    log.debug(f"Keywords: {json.dumps(kwds, **json_kwds)}")
    log.debug(f"Meta: {json.dumps(meta, **json_kwds)}")
    log.debug(f"Names: {json.dumps(names, **json_kwds)}")

    # Write control file.
    with open(os.path.join(directory, "input.nml"), "w") as fp:
        fp.write(utils.format_ferre_control_keywords(kwds))

    # Write data arrays.
    utils.write_data_file(flux[:, mask], os.path.join(directory, kwds["ffile"]))
    utils.write_data_file(sigma[:, mask], os.path.join(directory, kwds["erfile"]))

    # Write initial values.
    with open(os.path.join(directory, kwds["pfile"]), "w") as fp:
        for name, point in zip(names, initial_parameters):
            fp.write(utils.format_ferre_input_parameters(*point, name=name))    

    return parsed_kwds
    

def ferre(
        wavelength,
        flux,
        sigma,
        header_path,
        names=None,
        initial_parameters=None,
        frozen_parameters=None,
        interpolation_order=3,
        input_weights_path=None,
        input_lsf_shape_path=None,
        lsf_shape_flag=0,
        error_algorithm_flag=1,
        wavelength_interpolation_flag=0,
        optimization_algorithm_flag=3,
        continuum_flag=1,
        continuum_order=4,
        continuum_segment=None,
        continuum_reject=0.3,
        continuum_observations_flag=1,
        full_covariance=False,
        pca_project=False,
        pca_chi=False,
        n_threads=32,
        f_access=None,
        f_format=1,
        ferre_kwargs=None,
        directory=None,
        clean_up_on_exit=False,
        raise_exception_on_bad_outputs=False,
        **kwargs
    ):
    """
    Run FERRE on the given observations and return the parsed outputs.
    
    :param wavelength:
        An array of wavelength values for the observations. This should be one of:

        - a 1D array of shape `P` where P is the number of pixels, if all spectra are
          on the same wavelength grid
        - an array of shape `(N, P)` where `N` is the number of observations and `P` 
          is the number of pixels, if all spectra have the same number of pixels
        - a list of `N` arrays, where each array contains the number of pixels in 
          that observation
        
    :param flux:
        The observed flux values. This should be one of:

        - an array of shape `(N, P)` where `N` is the number of observations and `P`
          is the number of pixels, if all spectra have the same number of pixels
        - a list of `N` arrays, where each array has a size of the number of pixels in
          that observation.
        
    :param sigma:
        The uncertainty in the observed flux values. This should be one of:

        - an array of shape `(N, P)` where `N` is the number of observations and `P`
          is the number of pixels, if all spectra have the same number of pixels
        - a list of `N` arrays, where each array has a size of the number of pixels in
          that observation
        
    :param header_path:
        The path of the FERRE header file.
        
    :param initial_parameters: [optional]
        The initial parameters to start from. If `None` is given then this will revert
        to the mid-point of the grid for all observations. This should be an array of
        shape `(N, L)` where `N` is the number of observations and `L` is the number
        of dimensions in the FERRE grid supplied.

    :param frozen_parameters: [optional]
        A dictionary with parameter names (as per the header file) as keys, and either
        a boolean flag or a float as value. If boolean `True` is given for a parameter,
        then the value will be fixed at the initial value per spectrum. If a float is
        given then this value will supercede all initial values given, fixing the
        dimension for all input spectra regardless of the initial value.

    :param interpolation_order: [optional]
        Order of interpolation to use (default: 1, as per FERRE).
        This corresponds to the FERRE keyword `inter`.

        0. nearest neighbour
        1. linear
        2. quadratic Bezier
        3. cubic Bezier
        4. cubic splines

    :param input_weights_path: [optional]
        The location of a weight (or mask) file to apply to the pixels. This corresponds
        to the FERRE keyword `filterfile`.
    
    :para input_lsf_shape_path: [optional]
        The location of a file containing describing the line spread function to apply to
        the observations. This keyword is ignored if `lsf_shape_flag` is anything but 0.
        This corresponds to the FERRE keyword `lsffile`.
    
    :param lsf_shape_flag: [optional]
        A flag indicating what line spread convolution to perform. This should be one of:

        0. no LSF convolution (default)
        1. 1D (independent of wavelength), one and the same for all spectra
        2. 2D (a function of wavelength), one and the same for all
        3. 1D and Gaussian  (i.e. described by a single parameter, its width), one for all objects
        4. 2D and Gaussian, one for all objects
        11. 1D and particular for each spectrum
        12. 2D and particular for each spectrum
        13. 1D Gaussian, but particular for each spectrum
        14. 2D Gaussian and particular for each object.

        If `lsf_shape_flag` is anything but 0, then an `input_lsf_path` keyword argument
        will also be required, pointing to the location of the LSF file.

    :param error_algorithm_flag: [optional]
        Choice of algorithm to compute error bars (default: 1, as per FERRE).
        This corresponds to the FERRE keyword `errbar`.

        0. To adopt the distance from the solution at which $\chi^2$ = min($\chi^2$) + 1
        1. To invert the curvature matrix
        2. Perform numerical experiments injecting noise into the data

    :param wavelength_interpolation_flag: [optional]
        Flag to indicate what to do about wavelength interpolation (default: 0).
        This is not usually needed as the FERRE grids are computed on the resampled
        APOGEE grid. This corresponds to the FERRE keyword `winter`.

        0. No interpolation.
        1. Interpolate observations.
        2. The FERRE documentation says 'Interpolate fluxes', but it is not clear to the
           writer how that is any different from Option 1.

    :param optimization_algorithm_flag: [optional]
        Integer flag to indicate which optimization algorithm to use:

        1. Nelder-Mead
        2. Boender-Timmer-Rinnoy Kan
        3. Powell's truncated Newton method
        4. Nash's truncated Newton method

    :param continuum_flag: [optional]
        Choice of algorithm to use for continuum fitting (default: 1).
        This corresponds to the FERRE keyword `cont`, and is related to the
        FERRE keywords `ncont` and `rejectcont`.

        If `None` is supplied then no continuum keywords will be given to FERRE.

        1. Polynomial fitting using an iterative sigma clipping algrithm (set by
           `continuum_order` and `continuum_reject` keywords).
        2. Segmented normalization, where the data are split into `continuum_segment`
           segments, and the values in each are divided by their mean values.
        3. The input data are divided by a running mean computed with a window of
           `continuum_segment` pixels.
    
    :param continuum_order: [optional]
        The order of polynomial fitting to use, if `continuum_flag` is 1.
        This corresponds to the FERRE keyword `ncont`, if `continuum_flag` is 1.
        If `continuum_flag` is not 1, this keyword argument is ignored.
    
    :param continuum_segment: [optional]
        Either the number of segments to split the data into for performing normalization,
        (e.g., when `continuum_flag` = 2), or the window size to use when `continuum_flag`
        = 3. This corresponds to the FERRE keyword `ncont` if `continuum_flag` is 2 or 3.
        If `continuum_flag` is not 2 or 3, this keyword argument is ignored.

    :param continuum_reject: [optional]
        When using polynomial fitting with an iterative sigma clipping algorithm
        (`continuum_flag` = 1), this sets the relative error where data points will be
        excluded. Any data points with relative errors larger than `continuum_reject`
        will be excluded. This corresponds to the FERRE keyword `rejectcont`.
        If `continuum_flag` is not 1, this keyword argument is ignored.
    
    :param continuum_observations_flag: [optional]
        This corresponds to the FERRE keyword `obscont`. Nothing is written down in the
        FERRE documentation about this keyword.
    
    :param full_covariance: [optional]
        Return the full covariance matrix from FERRE (default: True). 
        This corresponds to the FERRE keyword `covprint`.
    
    :param pca_project: [optional]
        Use Principal Component Analysis to compress the spectra (default: False).
        This corresponds to the FERRE keyword `pcaproject`.
    
    :param pca_chi: [optional]
        Use Principal Component Analysis to compress the spectra when calculating the
        $\chi^2$ statistic. This corresponds to the FERRE keyword `pcachi`.
    
    :param n_threads: [optional]
        The number of threads to use for FERRE. This corresponds to the FERRE keyword
        `nthreads`.

    :param f_access: [optional]
        If `False`, load the entire grid into memory. If `True`, run the interpolation 
        without loading the entire grid into memory -- this is useful for small numbers 
        of interpolation. If `None` (default), automatically determine which is faster.
        This corresponds to the FERRE keyword `f_access`.
    
    :param f_format: [optional]
        File format of the FERRE grid: 0 (ASCII) or 1 (UNF format, default).
        This corresponds to the FERRE keyword `f_format`.

    :param ferre_kwargs: [optional]
        A dictionary of options to apply directly to FERRE, which will over-ride other
        settings supplied here, so use with caution.
    """

    # Create the temporary directory, if necessary.
    if directory is None:
        directory = mkdtemp(**kwargs.get("directory_kwds", {}))
        log.info(f"Created temporary directory {directory}")

    os.makedirs(directory, exist_ok=True)

    # Create a dictionary of all input keywords.
    input_kwds = {}
    for arg in getfullargspec(ferre).args:
        input_kwds[arg] = locals()[arg]

    wavelength, flux, sigma, mask, names, initial_parameters, kwds, meta = prepare_ferre(directory, input_kwds)

    execute_args = (directory, len(flux), kwds["offile"])
    
    if slurm_kwds:
        stdout, stderr = _execute_ferre_by_slurm(*execute_args, **slurm_kwds)
    else:
        stdout, stderr = _execute_ferre_by_subprocess(*execute_args)

    return parse_ferre_outputs(
        directory, 
        header_path,
        wavelength,
        flux,
        sigma,
        mask,
        names,
        initial_parameters,
        kwds,
        meta,
        clean_up_on_exit=clean_up_on_exit,
        raise_exception_on_bad_outputs=raise_exception_on_bad_outputs
    )


def parse_ferre_outputs(directory, header_path, wavelength, flux, sigma, mask, names, initial_parameters, kwds, meta,
    clean_up_on_exit=False,
    raise_exception_on_bad_outputs=False):

    # Get processing times.
    #processing_times = utils.get_processing_times(stdout, kwds["nthreads"])

    # Parse parameter outputs and uncertainties.
    try:
        output_names, param, param_err, output_meta = utils.read_output_parameter_file(
            os.path.join(directory, kwds["opfile"]),
            n_dimensions=kwds["ndim"],
            full_covariance=kwds["covprint"]
        )
    except:
        log.exception(f"Failed to load FERRE output parameter file at {os.path.join(directory, kwds['opfile'])}")
        raise
    
    # Parse flux outputs.
    try:
        model_flux = np.nan * np.ones_like(flux)
        model_flux[:, mask] = np.loadtxt(os.path.join(directory, kwds["offile"]))
    except:
        log.exception(f"Failed to load model flux from {os.path.join(directory, kwds['offile'])}:")
        raise

    if kwds.get("cont", None) is None:
        continuum = np.ones_like(model_flux)
    else:
        # Infer continuum.
        normalized_flux = np.nan * np.ones_like(flux)
        normalized_flux[:, mask] = np.loadtxt(os.path.join(directory, kwds["sffile"]))
        continuum = flux / normalized_flux

    meta.update(
        mask=mask,
        wavelength=wavelength,
        flux=flux,
        sigma=sigma,
        normalized_model_flux=model_flux,
        continuum=continuum
    )
    
    # Flag things.
    P, L = param.shape

    param_bitmask = bitmask.ParamBitMask()
    bitmask_flag = np.zeros((P, L), dtype=np.int64)
    
    grid_headers, *segment_headers = utils.read_ferre_headers(utils.expand_path(header_path))
    bad_lower = (grid_headers["LLIMITS"] + grid_headers["STEPS"]/8)
    bad_upper = (grid_headers["ULIMITS"] - grid_headers["STEPS"]/8)
    bitmask_flag[(param < bad_lower) | (param > bad_upper)] |= param_bitmask.get_value("GRIDEDGE_BAD")

    warn_lower = (grid_headers["LLIMITS"] + grid_headers["STEPS"])
    warn_upper = (grid_headers["ULIMITS"] - grid_headers["STEPS"])
    bitmask_flag[(param < warn_lower) | (param > warn_upper)] |= param_bitmask.get_value("GRIDEDGE_WARN")

    bitmask_flag[(param == -999) | (param_err < -0.01)] |= param_bitmask.get_value("FERRE_FAIL")

    # Check for any erroneous outputs
    if raise_exception_on_bad_outputs and np.any(bitmask_flag & param_bitmask.get_value("FERRE_FAIL")):
        v = bitmask_flag & param_bitmask.get_value("FERRE_FAIL")
        idx = np.where(np.any(bitmask_flag & param_bitmask.get_value("FERRE_FAIL"), axis=1))
        
        raise ValueError(f"FERRE returned all erroneous values for an entry: {idx} {v}")

    # Include processing times and bitmask etc.
    meta.update(
        bitmask_flag=bitmask_flag.tolist(), # .tolist() for postgresql encoding.
        #processing_times=processing_times,
        **output_meta
    )

    # need parameter names
    print(f"input names {names}")
    print(f"output_names: {output_names}")
    print(f"param: {param}")
    print(f"param_err: {param_err}")
    print(f"meta: {meta}")
    print(f"bitmask_flag: {bitmask_flag}")

    # Parse elapsed time.
    
    #print(f"times {processing_times}")

    if clean_up_on_exit:
        log.info(f"Removing directory {directory} and its contents")
        rmtree(directory)
    else:
        log.info(f"Leaving directory {directory} and its contents as clean_up_on_exit = {clean_up_on_exit}")

    return (param, param_err, meta)



_ferre_executable = "ferre.x"
# TODO: notti
_ferre_executable = "/uufs/chpc.utah.edu/common/home/sdss09/software/apogee/Linux/apogee/trunk/bin/ferre.x"



def _check_ferre_progress(output_flux_path):
    if os.path.exists(output_flux_path):
        return utils.line_count(output_flux_path)
    return 0

def _monitor_ferre_progress(process, total, output_flux_path, interval=30):
    stdout, stderr = ("", "")

    total_done, total_errors = (0, 0)
    with tqdm(total=total, desc="FERRE", unit="spectra") as pb:
        while total > total_done:
            if process is not None:  
                try:
                    _stdout, _stderr = process.communicate(timeout=interval)
                except subprocess.TimeoutExpired:
                    None
                else:
                    stdout += _stdout
                    stderr += _stderr

            n_done = _check_ferre_progress(output_flux_path)
            
            n_errors = stderr.lower().count("error")
        
            n_updated = n_done - total_done
            pb.update(n_updated)

            total_done = n_done
            total_errors = n_errors

            if n_errors > 0:
                pb.set_description(f"FERRE ({total_errors:.0f} errors)")
            pb.refresh()

    return (stdout, stderr, total_done, total_errors)


def _execute_ferre_by_slurm(directory, total, offile, interval=60, **kwargs):

    from slurm import queue as SlurmQueue

    label = "ferre"

    queue = SlurmQueue(verbose=True)
    queue.create(
        label=label,
        **kwargs
    )
    queue.append(_ferre_executable, dir=directory)
    queue.commit(hard=True, submit=True)

    log.info(f"Slurm job submitted with {queue.key} and keywords {kwargs} to run {_ferre_executable} in {directory}")
    log.info(f"\tJob directory: {queue.job_dir}")

    # 
    stdout_path = os.path.join(directory, f"{label}_01.o")
    stderr_path = os.path.join(directory, f"{label}_01.e")    
    output_flux_path = os.path.join(directory, offile)

    # Now we wait until the Slurm job is complete.
    t_init, t_to_start = (time(), None)
    while 100 > queue.get_percent_complete():

        sleep(interval)

        t = time() - t_init

        if not os.path.exists(stderr_path) and not os.path.exists(stdout_path):
            log.info(f"Waiting on job {queue.key} to start (elapsed: {t / 60:.0f} min)")
        else:
            log.info(f"Job in {queue.key} has started")
            
            total_done = 0
            with tqdm(total=total, desc="FERRE", unit="spectra") as pb:
                while total_done < total:
                    n_done = _check_ferre_progress(output_flux_path)            
                    pb.update(n_done - total_done)
                    total_done = n_done

                    pb.refresh()

                    sleep(interval)

            log.info("Finishing up.")

    with open(stdout_path, "r") as fp:
        stdout = fp.read()
    with open(stderr_path, "r") as fp:
        stderr = fp.read()

    return (stdout, stderr)



def _execute_ferre_by_subprocess(directory, total, offile, interval=1):

    # Execute in non-interactive mode.
    t_0 = time()
    try:
        process = subprocess.Popen(
            [_ferre_executable],
            cwd=directory,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            encoding="utf-8",
            close_fds="posix" in sys.builtin_module_names
        )

    except subprocess.CalledProcessError:
        log.exception(f"Exception when calling FERRE in {directory}")
        raise

    else:
        output_flux_path = os.path.join(directory, offile)
        stdout, stderr, total_done, total_errors = _monitor_ferre_progress(process, total, output_flux_path)

        # Get final processing
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            None
    
    return (stdout, stderr) 