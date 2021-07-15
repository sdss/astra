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
from tempfile import mkdtemp
from time import sleep, time
from tqdm import tqdm

from astra.utils import log
from astra.contrib.ferre import utils

# Cross-check
# /uufs/chpc.utah.edu/common/home/sdss50/dr17/apogee/spectro/aspcap/dr17/synspec/bundle_apo25m/apo25m_003/ferre/elem_K

def ferre(
        wavelength,
        flux,
        sigma,
        header_path,
        names=None,
        initial_parameters=None,
        frozen_parameters=None,
        interpolation_order=1,
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
        full_covariance=True,
        pca_project=False,
        pca_chi=False,
        n_threads=1,
        f_access=None,
        f_format=1,
        ferre_kwds=None,
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

    :param ferre_kwds: [optional]
        A dictionary of options to apply directly to FERRE, which will over-ride other
        settings supplied here, so use with caution.
    """

    # Create a dictionary of all input keywords.
    input_kwds = {}
    for arg in getfullargspec(ferre).args:
        input_kwds[arg] = locals()[arg]
    
    # Parse and validate parameters.
    wavelength, flux, sigma, mask, names, initial_parameters, kwds, meta = utils.parse_ferre_inputs(**input_kwds)

    # Create the temporary directory, if necessary.
    directory = kwargs.get("directory", None)
    if directory is None:
        directory = mkdtemp(**kwargs.get("directory_kwds", {}))
        log.info(f"Created temporary directory {directory}")
    
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

    # Execute in non-interactive mode.
    t_0 = time()
    try:
        process = subprocess.Popen(
            ["ferre.x"],
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
        interval = 1
        total = len(flux)
        stdout, stderr = ("", "")
        output_flux_path = os.path.join(directory, kwds["offile"])

        total_done, total_errors = (0, 0)
        with tqdm(total=total, desc="FERRE", unit="spectra") as pb:
            while total > total_done:
                try:
                    _stdout, _stderr = process.communicate(timeout=interval)
                except subprocess.TimeoutExpired:
                    None
                else:
                    stdout += _stdout
                    stderr += _stderr

                if os.path.exists(output_flux_path):
                    n_done = utils.line_count(output_flux_path)
                else:
                    n_done = 0
                
                n_errors = stderr.lower().count("error")
            
                n_updated = n_done - total_done
                pb.update(n_updated)

                total_done = n_done
                total_errors = n_errors

                if n_errors > 0:
                    pb.set_description(f"FERRE ({total_errors:.0f} errors)")
                pb.refresh()

        # Get final processing
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            processing_times = None
        else:
            processing_times = utils.get_processing_times(stdout, kwds["nthreads"])
            

    # Parse parameter outputs and uncertainties.
    output_names, param, param_err, output_meta = utils.read_output_parameter_file(
        os.path.join(directory, kwds["opfile"]),
        n_dimensions=kwds["ndim"],
        full_covariance=kwds["covprint"]
    )
    
    meta.update(
        processing_times=processing_times,
        **output_meta
    )

    # need parameter names

    print(f"input names {names}")
    print(f"output_names: {output_names}")
    print(f"param: {param}")
    print(f"param_err: {param_err}")
    print(f"meta: {meta}")

    # Parse elapsed time.
    
    print(f"times {processing_times}")

    # Parse flux outputs.
    # exclude frozen things.
    return (param, param_err, meta)




