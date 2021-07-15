
import datetime
import os
import numpy as np
import re
import subprocess

from collections import OrderedDict
from inspect import getfullargspec


from astra.utils import log

expand_path = lambda path: os.path.expandvars(os.path.expanduser(path))



# TODO: Put elsewhere
def _grid_mid_point(headers):
    return np.mean(
        np.vstack([
            headers["LLIMITS"],
            headers["ULIMITS"]
        ]),
        axis=0
    )



def _parse_pca_chi(pca_chi, **kwargs):
    return dict(pcachi=int(pca_chi))

def _parse_pca_project(pca_project, **kwargs):
    return dict(pcaproject=int(pca_project))

def _parse_full_covariance(full_covariance, **kwargs):
    return dict(covprint=int(full_covariance))

def _parse_optimization_algorithm_flag(optimization_algorithm_flag, **kwargs):
    available = (1, 2, 3, 4)
    optimization_algorithm_flag = int(optimization_algorithm_flag)
    if optimization_algorithm_flag not in available:
        raise ValueError(f"optimization_algorithm_flag must be one of {available}")
    return dict(algor=optimization_algorithm_flag)


def _parse_interpolation_order(interpolation_order, **kwargs):
    available = {
        0: "nearest neighbour",
        1: "linear",
        2: "quadratic Bezier",
        3: "cubic Bezier",
        4: "cubic splines"
    }
    interpolation_order = int(interpolation_order)
    if interpolation_order not in available:
        raise ValueError(f"interpolation_order must be one of {tuple(list(available.keys()))}")
    return dict(inter=interpolation_order)


def _parse_error_algorithm_flag(error_algorithm_flag, **kwargs):
    available = {
        0: "adopt distance from the solution at which \chi^2 = min(\chi^2) + 1",
        1: "invert the curvature matrix",
        2: "perform numerical experiments injecting noise into the data"
    }
    error_algorithm_flag = int(error_algorithm_flag)    
    if error_algorithm_flag not in available:
        raise ValueError(f"error_algorithm_flag must be one of {tuple(list(available.keys()))}")
    return dict(errbar=error_algorithm_flag)


def _parse_input_weights_path(input_weights_path, **kwargs):
    if input_weights_path is not None:
        input_weights_path = expand_path(input_weights_path)
        if not os.path.exists(input_weights_path):
            raise ValueError(f"the input_weights_path does not exist: {input_weights_path}")
        return dict(filterfile=input_weights_path)
    
def _parse_lsf_shape_flag_and_input_lsf_shape_path(input_lsf_shape_path, lsf_shape_flag, **kwargs):
    available = (0, 1, 2, 3, 4, 11, 12, 13, 14)
    if lsf_shape_flag is not None:    
        lsf_shape_flag = int(lsf_shape_flag)
        if lsf_shape_flag not in available:
            raise ValueError(f"lsf_shape_flag must be one of {available}")

        if lsf_shape_flag != 0:
            if input_lsf_shape_path is None:
                raise ValueError(f"lsf_shape_flag is not 0, so an `input_lsf_shape_path` is needed")

            input_lsf_shape_path = expand_path(input_lsf_shape_path)
            if not os.path.exists(input_lsf_shape_path):
                raise ValueError(f"input_lsf_shape_path does not exist: {input_lsf_shape_path}")

            return dict(
                lsf=lsf_shape_flag,
                lsffile=input_lsf_shape_path
            )
    
    return None


def _parse_wavelength_interpolation_flag(wavelength_interpolation_flag, wavelength, **kwargs):
    available = (0, 1, 2)
    wavelength_interpolation_flag = int(wavelength_interpolation_flag or 0)
    if wavelength_interpolation_flag not in available:
        raise ValueError(f"wavelength_interpolation_flag must be one of {available}")
    
    if wavelength_interpolation_flag > 0:
        if wavelength is None:
            raise ValueError("if wavelength_interpolation_flag != 0 then wavelength must be given")
        return dict(winter=wavelength_interpolation_flag)
    
    return None

def _parse_header_path(header_path, **kwargs):
    header_path = expand_path(header_path)
    if not os.path.exists(header_path):
        raise ValueError(f"header_path does not exist: {header_path}")
    return dict([("synthfile(1)", header_path)])

def _parse_n_threads(n_threads, **kwargs):
    return dict(nthreads=int(n_threads))

def _parse_f_access(f_access, **kwargs):
    if f_access is None:
        # TODO: Figure out which is faster. For now default to load everything.
        f_access = False
    return dict(f_access=int(f_access))

def _parse_f_format(f_format, **kwargs):
    available = (0, 1)
    f_format = int(f_format)
    if f_format not in available:
        raise ValueError(f"f_format must be one of {available}")
    return dict(f_format=f_format)


def _parse_continuum_args(
        continuum_flag,
        continuum_order,
        continuum_segment,
        continuum_reject,
        continuum_observations_flag,
        **kwargs
    ):
    kwds = dict()
    if continuum_flag is not None:
        continuum_flag = int(continuum_flag)
        available = (0, 1, 2, 3)
        if continuum_flag not in available:
            raise ValueError(f"continuum_flag must be one of {available}")

        if continuum_flag == 1:
            # need continuum_order and continuum_reject
            if continuum_order is None:
                raise ValueError("continuum_order is required if continuum_flag == 1")
            if continuum_reject is None:
                raise ValueError("continuum_reject is required if continuum_flag == 1")
                
            kwds.update(
                cont=continuum_flag,
                rejectcont=float(continuum_reject),
                ncont=int(continuum_order)
            )
        
        elif continuum_flag in (2, 3):
            if continuum_segment is None:
                raise ValueError("continuum_segment is required if continuum_flag is 2 or 3")
            
            if continuum_flag == 2:
                continuum_segment = int(continuum_segment)
            elif continuum_flag == 3:
                continuum_segment = float(continuum_segment)
            
            if continuum_segment < 1:
                raise ValueError(f"continuum_segment must be a positive value")
            
            kwds.update(
                cont=continuum_flag,
                ncont=continuum_segment
            )
        
    if continuum_observations_flag is not None:
        kwds.update(obscont=int(continuum_observations_flag))
    
    return kwds


def _parse_ferre_kwds(ferre_kwds, **kwargs):
    return (ferre_kwds or dict())


def _parse_wavelength_and_flux_and_sigma(
        wavelength,
        flux,
        sigma,
        wavelength_interpolation_flag,
        bad_pixel_value=1.0,
        bad_sigma_value=1e6,
        **kwargs
    ):

    # Check flux and sigma
    try:
        flux = np.atleast_2d(np.vstack(flux))
    except ValueError:
        # the number of pixels changes per observation.
        P_flux = list(map(len, flux))
        N_flux = len(flux)

    else:
        N_flux, P_flux = flux.shape
        P_flux = np.atleast_1d(P_flux)
    
    try:
        sigma = np.atleast_2d(np.vstack(sigma))
    except ValueError:
        P_sigma = list(map(len, sigma))
        N_sigma = len(sigma)
    else:
        N_sigma, P_sigma = sigma.shape
        P_sigma = np.atleast_1d(P_sigma)

    if N_flux != N_sigma:
        raise ValueError(f"the number of flux arrays and sigma arrays do not match ({N_flux} != {N_sigma})")
    
    if type(P_flux) != type(P_sigma):
        raise ValueError(f"the flux and sigma arrays are incompatible ({type(flux)} and {type(sigma)})")
    
    if not np.all(P_flux == P_sigma):
        N_mismatches = np.sum(P_flux != P_sigma)
        raise ValueError(f"different length arrays for flux and sigmas: {N_mismatches} mis-matches with {P_flux} and {P_sigma}")

    # TODO: This will cause problems when the number of pixels change per observation.
    bad = ~np.isfinite(flux) + ~np.isfinite(sigma) + (sigma == 0)
    flux[bad] = bad_pixel_value
    sigma[bad] = bad_sigma_value

    if wavelength_interpolation_flag > 0:
        # Check the wavelength values.
        raise NotImplementedError("check wavelength shapes match flux and sigma shapes")
    
    assert wavelength is not None
    wavelength = np.atleast_1d(wavelength)

    return (wavelength, flux, sigma)




def _default_ferre_kwds(**kwargs):
    return {
        "pfile": "parameters.input",
        "wfile": "wavelengths.input",
        "ffile": "flux.input",
        "erfile": "uncertainties.input",
        "opfile": "parameters.output",
        "offile": "flux.output",
        "sffile": "normalized_flux.output"
    }


def _parse_names_and_initial_and_frozen_parameters(
        names,
        initial_parameters,
        frozen_parameters,
        headers,
        flux,
        **kwargs
    ):

    # Read the labels from the first header path
    parameter_names = headers["LABEL"]

    # Need the number of spectra, which we will take from the flux array.
    N = len(flux)
    mid_point = _grid_mid_point(headers)
    parsed_initial_parameters = np.tile(mid_point, N).reshape((N, -1))    

    compare_parameter_names = list(map(sanitise_parameter_name, parameter_names))
        
    if initial_parameters is not None:
        for i, (parameter_name, values) in enumerate(initial_parameters.items()):
            try:
                index = compare_parameter_names.index(sanitise_parameter_name(parameter_name))
            except ValueError:
                log.warning(f"Ignoring initial parameters for {parameter_name} as they are not in {parameter_names}")
            else:
                parsed_initial_parameters[:, index] = values
        
    kwds = dict()
    frozen_parameters = (frozen_parameters or dict())
    if frozen_parameters:
        # Ensure we have a dict-like thing.
        if isinstance(frozen_parameters, (list, tuple, np.ndarray)):
            frozen_parameters = { sanitise_parameter_name(k): True for k in frozen_parameters }
        elif isinstance(frozen_parameters, dict):
            # Exclude things that have boolean False.
            frozen_parameters = { 
                sanitise_parameter_name(k): v for k, v in frozen_parameters.items() \
                if not (isinstance(v, bool) and not v)
            }
        else:
            raise TypeError(f"frozen_parameters must be list-like or dict-like")
        
        unknown_parameters = set(frozen_parameters).difference(compare_parameter_names)
        if unknown_parameters:
            raise ValueError(f"unknown parameter(s): {unknown_parameters} (available: {parameter_names})")

        indices = [
            i for i, pn in enumerate(compare_parameter_names, start=1) if pn not in frozen_parameters
        ]

        if len(indices) == 0:
            raise ValueError(f"all parameters frozen?!")

        # Over-ride initial values with the frozen ones if given.
        for parameter_name, value in frozen_parameters.items():
            if not isinstance(value, bool):
                log.debug(f"Over-writing initial values for {parameter_name} with frozen value of {value}")
                zero_index = compare_parameter_names.index(parameter_name)
                parsed_initial_parameters[:, zero_index] = value
    else:
        # No frozen parameters.
        indices = 1 + np.arange(len(parameter_names), dtype=int)
    
    # Build a frozen parameters dict for result metadata.
    parsed_frozen_parameters = { pn: (pn in frozen_parameters) for pn in compare_parameter_names }

    L = len(indices)
    kwds.update(
        ndim=headers["N_OF_DIM"],
        nov=L,
        indv=" ".join([f"{i:.0f}" for i in indices]),
        # We will always provide an initial guess, even if it is the grid mid point.
        init=0,
        indini=" ".join(["1"] * L)
    )

    # Now deal with names.
    if names is None:
        names = [f"{i:.0f}" for i in range(len(parsed_initial_parameters))]
    else:
        if len(names) != len(parsed_initial_parameters):
            raise ValueError(f"names and initial parameters does not match ({len(names)} != {len(parsed_initial_parameters)})")

    return (kwds, names, parsed_initial_parameters, parsed_frozen_parameters)








def parse_ferre_inputs(**kwargs):
        
    parsed_kwds = {}

    headers, *segment_headers = read_ferre_headers(expand_path(kwargs["header_path"]))

    # These two have different outputs than the rest.
    wavelength, flux, sigma = _parse_wavelength_and_flux_and_sigma(**kwargs)

    # Build wavelength arrays to generate a mask.
    model_wavelengths = tuple(map(wavelength_array_from_ferre_header, segment_headers))

    # Build from first wavelength array
    mask = np.zeros(wavelength.shape[1], dtype=bool)
    for model_wavelength in model_wavelengths:
        # TODO: Building wavelength mask off just the first wavelength array. Assuming all have the same wavelength array.
        s_index, e_index = wavelength[0].searchsorted(model_wavelength[[0, -1]])
        mask[s_index:e_index + 1] = True
        
    _kwds, names, initial_parameters, parsed_frozen_parameters = _parse_names_and_initial_and_frozen_parameters(headers=headers, **kwargs)
    parsed_kwds.update(_kwds)

    parsers = (
        _default_ferre_kwds,
        _parse_pca_chi,
        _parse_pca_project,
        _parse_full_covariance,
        _parse_optimization_algorithm_flag,
        _parse_interpolation_order,
        _parse_error_algorithm_flag,
        _parse_input_weights_path,
        _parse_lsf_shape_flag_and_input_lsf_shape_path,
        _parse_wavelength_interpolation_flag,
        _parse_header_path,
        _parse_n_threads,
        _parse_f_access,
        _parse_f_format,
        _parse_continuum_args,
        _parse_ferre_kwds,
    )

    for parser in parsers:
        _kwds = parser(**kwargs)
        if _kwds: parsed_kwds.update(_kwds)

    # Create a metadata dict for results.
    meta = dict(
        parameter_names=headers["LABEL"],
        frozen_parameters=parsed_frozen_parameters,
        initial_parameters=initial_parameters
    )
    return (wavelength, flux, sigma, mask, names, initial_parameters, parsed_kwds, meta)

    

def non_blocking_pipe_read(stream, queue):
    """ 
    A non-blocking and non-destructive stream reader for long-running interactive jobs. 
    
    :param stream:
        The stream to read (e.g., process.stderr).
    
    :param queue:
        The multiprocessing queue to put the output from the stream to.        
    """

    thread = threading.currentThread()
    while getattr(thread, "needed", True):
        f = stream.readline()
        queue.put(f)

    return None


sanitise_parameter_name = lambda pn: pn.lower().strip().replace(" ", "_")
desanitise_parameter_name = lambda pn: pn.upper().replace("_", " ")
    

def line_count(filename):
    return int(subprocess.check_output(['wc', '-l', filename]).split()[0])


def wavelength_array_from_ferre_header(header):
    wave = header["WAVE"][0] + np.arange(header["NPIX"]) * header["WAVE"][1]
    if header["LOGW"] == 1:
        wave = 10**wave
    elif header["LOGW"] == 2:
        wave = np.exp(wave)

    return wave


def write_data_file(data, path, fmt="%.4e", footer="\n", **kwargs):
    return np.savetxt(
        path,
        data,
        fmt=fmt,
        footer=footer,
        **kwargs
    )


def safe_read_header(headers, keys):
    if not isinstance(keys, (tuple, list)):
        keys = [keys]
    
    for key in keys:
        try:
            return headers[key]
        except KeyError:
            continue
    
    else:
        raise KeyError(f"no header keyword found among {keys}")

        


def read_ferre_header(fp):
    """ 
    Read a FERRE library header into a dictionary.

    This functionality was originally written by Jon Holtzmann.

    :param fp:
        A file pointer to a FERRE header file.
    """

    header = dict()
    for i, line in enumerate(fp):
        if line.startswith(" /"):
            break

        if "=" in line:
            key, value = line.split(" = ")
            
            # Check values.
            if value.lstrip()[0] != "'":
                # Treat as numerical values.
                values = re.findall(
                    "[+|-|\d|\.|e|-]+",
                    value
                )

                dtype = float if "." in value else int

                if len(values) == 1:
                    value = dtype(values[0])
                else:
                    value = np.array(values, dtype=dtype)
            
            else:
                value = value.strip(" '\n")
                
            match = re.match(r'\s+(?P<key>\w+)\((?P<index>\d+)\)', key)
            if match:
                match = match.groupdict()
                key = match["key"]
                header.setdefault(key, [])
                # TODO: assuming header information is in order
                header[key].append(value)
            else:
                key = key.strip()
                header[key] = value

    # Put in upper grid limits as a 'value-added' header.
    header["ULIMITS"] = header["LLIMITS"] + header["STEPS"] * header["N_P"]

    return header


def read_ferre_headers(path):
    """ 
    Read a full FERRE library header with multi-extensions.
    :param path:
        The path of a FERRE header file.
    Returns:
       libstr0, libstr : first header, then list of extension headers; headers returned as dictionaries
    """

    try:
        with open(path, "r") as fp:

            headers = [read_ferre_header(fp)]
            headers[0]["PATH"] = path

            for i in range(headers[0].get("MULTI", 0)):
                headers.append(read_ferre_header(fp))    
    
    except:
        raise
    
    else:
        return headers


def get_grid_boundaries(ferre_grid_header_files, **kwargs):
    """
    Get the grid boundaries for the given FERRE grid header files.

    :param ferre_grid_header_files:
        A list of FERRE grid paths.
    
    :returns:
        A dictionary with FERRE grid paths as keys, and each value contains
        a dictionary of label names as keys and a tuple of (lower, upper)
        boundary values.
    """
    boundaries = {}

    for path in ferre_grid_header_files:
        expanded_path = os.path.expandvars(path)
        primary_header, *headers = read_ferre_headers(expanded_path)

        args = (
            primary_header["LABEL"],
            primary_header["LLIMITS"],
            primary_header["ULIMITS"]
        )
        boundaries[expanded_path] = {
            label_name: (lower, upper) for label_name, lower, upper in zip(*args)
        }

    return boundaries
    

def parse_grid_information(header_paths):
    """
    Parse the parameter limits of a pre-computed grid (e.g., in effective temperature) from grid header paths provided,
    and other possibly relevant information (e.g., telescope of the LSF model).

    :param header_paths:
        A list of paths that store information about pre-computed grids.
    
    :returns:
        A dictionary with paths as keys, and the value of the dictionary is a three-length tuple containing (1) metadata, (2) lower limits in parameters, (2) upper limits in parameters.
    """

    grids = {}
    for header_path in header_paths:
        
        full_header_path = os.path.expandvars(header_path)

        try:
            headers = read_ferre_headers(full_header_path)
            meta = parse_header_path(full_header_path)

        except:
            raise
    
        else:
            # Get grid limits.
            grids[header_path] = (meta, list(headers[0]["LLIMITS"]), list(headers[0]["ULIMITS"]))

    return grids
        


def yield_suitable_grids(grid_info, mean_fiber, teff, logg, metals, telescope, **kwargs):
    """
    Yield suitable FERRE grids given header information from an observation and a dictionary of grid limits.
    
    :param grid_info:
        A dictionary containing header paths as keys, and a three-length tuple as values: (1) metadata, (2) lower limits, (3) upper limits.
        This is the expected output from `parse_grid_information`.
    
    :param mean_fiber:
        The mean fiber number of observations.
    
    :param teff:
        An initial guess of the effective temperature.
    
    :param logg:
        An initial guess of the surface gravity.
    
    :param metals:
        An initial guess of the metallicity.
    
    :returns:
        A generator that yields two-length tuples containing header path, and metadata..
    """

    # Figure out which grids are suitable.
    lsf_grid = get_lsf_grid_name(int(np.round(mean_fiber)))

    point = np.array([metals, logg, teff])
    P = point.size
    
    for header_path, (meta, lower_limits, upper_limits) in grid_info.items():

        #print(meta["lsf"], lsf_grid, telescope, meta["lsf_telescope_model"], header_path)
        # Match star to LSF fiber number model (a, b, c, d) and telescope model (apo25m/lco25m).
        # TODO: This is a very APOGEE-specific thing and perhaps should be moved elsewhere.
        if meta["lsf"] != lsf_grid or telescope != meta["lsf_telescope_model"]:
            continue

        # We will take the RV parameters as the initial parameters. 
        # Check to see if they are within bounds of the grid.
        if np.all(point >= lower_limits[-P:]) and np.all(point <= upper_limits[-P:]):
            yield (header_path, meta)


def get_lsf_grid_name(fibre_number):
    """
    Return the appropriate LSF name (a, b, c, or d) to use, given a mean fiber number.

    :param fiber_number:
        The mean fiber number of observations.
    
    :returns:
        A one-length string describing which LSF grid to use ('a', 'b', 'c', or 'd').
    """
    if 50 >= fibre_number >= 1:
        return "d"
    if 145 >= fibre_number > 50:
        return "c"
    if 245 >= fibre_number > 145:
        return "b"
    if 300 >= fibre_number > 245:
        return "a"


def parse_header_path(header_path):
    """
    Parse the path of a header file and return a dictionary of relevant parameters.

    :param header_path:
        The path of a grid header file.
    
    :returns:
        A dictionary of keywords that are relevant to running FERRE tasks.
    """

    *_, radiative_transfer_code, model_photospheres, isotopes, folder, basename = header_path.split("/")

    parts = basename.split("_")
    # p_apst{gd}{spectral_type}_{date}_lsf{lsf}_{aspcap}_012_075
    _ = 4
    gd, spectral_type = (parts[1][_], parts[1][_ + 1:])
    date_str = parts[2]
    year, month, day = (2000 + int(date_str[:2]), int(date_str[2:4]), int(date_str[4:]))
    lsf = parts[3][3]
    lsf_telescope_model = "lco25m" if parts[3][4:] == "s" else "apo25m"

    aspcap = parts[4]

    is_giant_grid = gd == "g"
    
    return dict(
        radiative_transfer_code=radiative_transfer_code,
        model_photospheres=model_photospheres,
        isotopes=isotopes,
        gd=gd,
        lsf_telescope_model=lsf_telescope_model,
        spectral_type=spectral_type,
        grid_creation_date=datetime.date(year, month, day),
        lsf=lsf,
        aspcap=aspcap,
    )




def approximate_log10_microturbulence(log_g):
    """
    Approximate the log10(microturbulent velocity) given the surface gravity.

    :param log_g:
        The surface gravity.
    
    :returns:
        The log base-10 of the microturbulent velocity, vt: log_10(vt).
    """

    coeffs = np.array([0.372160, -0.090531, -0.000802, 0.001263, -0.027321])
    # I checked with Holtz on this microturbulence relation because last term is not used.
    DM = np.array([1, log_g, log_g**2, log_g**3, 0])
    return DM @ coeffs



def read_output_parameter_file(path, n_dimensions, full_covariance, **kwargs):
    """
    Three more columns follow, giving the fraction of photometric data
    points (useful when multiple grids combining spectroscopy and photometry are used),
    the average log(S/N)2
    for the spectrum, and the logarithm of the reduced χ^2 for the fit.

    Additional columns with the covariance matrix of the errors can be output setting to 1 the
    keyword COVPRINT.
    """

    names = np.loadtxt(path, usecols=(0, ), dtype=str)
    
    N_cols = 2 * n_dimensions
    if full_covariance:
        N_cols += 3 + n_dimensions**2

    results = np.atleast_2d(np.loadtxt(path, usecols=1 + np.arange(N_cols)))

    param = results[:, 0:n_dimensions]
    param_err = results[:, n_dimensions:2*n_dimensions]

    meta = dict(
        frac_phot_data_points=results[:, -3],
        log_snr_sq=results[:, -2],
        log_chisq_fit=results[:, -1],
    #    cov=cov,
    )
    if full_covariance:
        cov = results[:, 2*n_dimensions+3:2*n_dimensions+3+n_dimensions**2]
        cov = cov.reshape((-1, n_dimensions, n_dimensions))
        meta.update(cov=cov)

    return (names, param, param_err, meta)





def format_ferre_control_keywords(ferre_kwds):
    r"""
    Format control keywords for FERRE to digest.

    :param ferre_kwds:
        A dictionary of FERRE-recognized keywords, in a suitable order.

    :returns:
        String contents for a FERRE input file.
    """

    preferred_order = (
        "ndim",
        "nov",
        "indv",
        "synthfile(1)",
        "pfile",
        "ffile",
        "erfile",
        "opfile",
        "offile",
        "obscont",
        "inter",
        "errbar",
        "nruns",
        "init",
        "indini",
        "winter",
        "algor",
        "lsf",
        "nthreads",
        "covprint",
        "pcachi",
        "f_format",
        "f_access",
    )

    contents = "&LISTA\n"
    remaining_keys = set(ferre_kwds).difference(preferred_order)
    keys = list(preferred_order) + list(remaining_keys)

    for key in keys:
        if key in ferre_kwds:
            value = ferre_kwds[key]
            if isinstance(value, str) and key not in ("indv", "indini"):
                value = f"'{value}'"
            else:
                value = f"{value}"
            contents += f"{key.upper()} = {value}\n"

    contents += "/\n"
    return contents





# Note: this old_format_ferre_input_parameters certainly will match the IPF files on the SDSS SAS,
#       but are giving me FERRE problems right now. Check format later on.
def format_ferre_input_parameters(*p, name="dummy"):
    r"""
    Format input parameters for FERRE to digest.

    :returns:
        String contents for a FERRE input parameter file.
    """
    contents = f"{name:40s}"
    for each in p:
        contents += f"{each:12.3f} "
    contents += "\n"
    return contents




def get_processing_times(stdout, n_threads):
    """
    Get the time taken to analyse spectra and estimate the initial load time.

    :param stdout: (optional)
        The standard output from FERRE.
    """

    if stdout is None or stdout == "":
        return None

    matches = re.findall('ellapsed time:\s+[{0-9}|.]+', stdout)
    load_time, *elapsed_time = [float(match.split()[-1]) for match in matches]

    # Offset load time.
    elapsed_time = np.array(elapsed_time) - load_time

    # Account for number of threads.
    O = n_threads - (elapsed_time.size % n_threads)
    A = np.hstack([elapsed_time, np.zeros(O)]).reshape((-1, n_threads))
    time_per_spectrum = np.hstack([elapsed_time[:n_threads], np.diff(A, axis=0).flatten()[:-O]])

    object_indices = [int(index.split()[-1]) for index in re.findall('next object #\s+[{0-9}]+', stdout)]

    # Sort.
    idx = np.argsort(object_indices)
    time_per_spectrum = time_per_spectrum[idx]
    
    return dict(
        load_time=load_time,
        elapsed_time=elapsed_time,
        indices=idx,
        time_per_ordered_spectrum=time_per_spectrum
    )



def get_abundance_keywords(element, header_label_names):
    """
    Return a dictionary of task parameters given a chemical element. These are adopted from DR16.

    :param element:
        The chemical element to measure.

    :param header_label_names:
        The list of label names in the FERRE header file.
    """

    # These can be inferred from running the following command on the SAS:
    # cd /uufs/chpc.utah.edu/common/home/sdss50/dr16/apogee/spectro/aspcap/r12/l33/apo25m/cal_all_apo25m007/ferre
    # egrep 'INDV|TIE|FILTERFILE' */input.nml
    
    controls = {
        "Al": {
            "INDV_LABEL": ('METALS', ),
            "TIES": [
                ('C', 0, -1),
                ('N', 0, -1),
                ('O Mg Si S Ca Ti', 0, -1)
            ]
        },
        "Ca": {
            "INDV_LABEL": ('O Mg Si S Ca Ti', ),
        },
        "Ce": {
            "INDV_LABEL": ('METALS', ),
        },
        "CI": {
            "INDV_LABEL": ('C', ),
        },
        "C": {
            "INDV_LABEL": ('C', ),
        },
        "CN": {
            "INDV_LABEL": ('C', 'O Mg Si S Ca Ti', ),
        },
        "Co": {
            "INDV_LABEL": ('METALS', ),
            "TIES": [
                ('C', 0, -1),
                ('N', 0, -1),
                ('O Mg Si S Ca Ti', 0, -1)
            ]    
        },
        "Cr": {
            "INDV_LABEL": ('METALS', ),
            "TIES": [
                ('C', 0, -1),
                ('N', 0, -1),
                ('O Mg Si S Ca Ti', 0, -1)
            ]
        },
        "Cu": {
            "INDV_LABEL": ('METALS', ),
            "TIES": [
                ('C', 0, -1),
                ('N', 0, -1),
                ('O Mg Si S Ca Ti', 0, -1)
            ]
        },
        "Fe": {
            "INDV_LABEL": ('METALS', ),
            "TIES": [
                ('C', 0, -1),
                ('N', 0, -1),
                ('O Mg Si S Ca Ti', 0, -1)
            ]
        },
        "Ge": {
            "INDV_LABEL": ('METALS', ),
            "TIES": [
                ('C', 0, -1),
                ('N', 0, -1),
                ('O Mg Si S Ca Ti', 0, -1)
            ]
        },
        "K": {
            "INDV_LABEL": ('METALS', ),
            "TIES": [
                ('C', 0, -1),
                ('N', 0, -1),
                ('O Mg Si S Ca Ti', 0, -1)
            ]
        },
        "Mg": {
            "INDV_LABEL": ('O Mg Si S Ca Ti', ),
        },
        "Mn": {
            "INDV_LABEL": ('METALS', ),
            "TIES": [
                ('C', 0, -1),
                ('N', 0, -1),
                ('O Mg Si S Ca Ti', 0, -1)
            ]
        },
        "Na": {
            "INDV_LABEL": ('METALS', ),
            "TIES": [
                ('C', 0, -1),
                ('N', 0, -1),
                ('O Mg Si S Ca Ti', 0, -1)
            ]
        },
        "Nd": {
            "INDV_LABEL": ('METALS', ),
            "TIES": [
                ('C', 0, -1),
                ('N', 0, -1),
                ('O Mg Si S Ca Ti', 0, -1)
            ]
        },
        "Ni": {
            "INDV_LABEL": ('METALS', ),
            "TIES": [
                ('C', 0, -1),
                ('N', 0, -1),
                ('O Mg Si S Ca Ti', 0, -1)
            ]
        },
        "N": {
            "INDV_LABEL": ('N', ),
        },
        "O": {
            "INDV_LABEL": ('O Mg Si S Ca Ti', ),
        },
        "P": {
            "INDV_LABEL": ('METALS', ),
            "TIES": [
                ('C', 0, -1),
                ('N', 0, -1),
                ('O Mg Si S Ca Ti', 0, -1)
            ]
        },
        "Rb": {
            "INDV_LABEL": ('METALS', ),
            "TIES": [
                ('C', 0, -1),
                ('N', 0, -1),
                ('O Mg Si S Ca Ti', 0, -1)
            ]
        },
        "Si": {
            "INDV_LABEL": ('O Mg Si S Ca Ti', ),
        },
        "S": {
            "INDV_LABEL": ('O Mg Si S Ca Ti', ),
        },
        "TiII": {
            "INDV_LABEL": ('O Mg Si S Ca Ti', ),
        },
        "Ti": {
            "INDV_LABEL": ('O Mg Si S Ca Ti', ),
        },
        "V": {
            "INDV_LABEL": ('METALS', ),
            "TIES": [
                ('C', 0, -1),
                ('N', 0, -1),
                ('O Mg Si S Ca Ti', 0, -1)
            ]
        },
        "Yb": {
            "INDV_LABEL": ('METALS', ),
            "TIES": [
                ('C', 0, -1),
                ('N', 0, -1),
                ('O Mg Si S Ca Ti', 0, -1)
            ]
        }
    }

    def get_header_index(label):
        # FERRE uses 1-indexing and Python uses 0-indexing.
        return 1 + header_label_names.index(label)

    try:
        c = controls[element]
    except:
        raise ValueError(f"no abundance controls known for element '{element}' (available: {tuple(controls.keys())}")

    indv = [get_header_index(label) for label in c["INDV_LABEL"]]
    ties = c.get("TIES", [])

    ferre_kwds = {
        # We don't pass INDV here because this will be determined from the
        # 'frozen_<param>' arguments to the FerreGivenSDSSApStarFile tasks
        #"INDV": [get_header_index(label) for label in c["INDV_LABEL"]],
        "NTIE": len(ties),
        "TYPETIE": 1
    }
    for i, (tie_label, ttie0, ttie) in enumerate(ties, start=1):
        ferre_kwds.update({
            f"INDTIE({i:.0f})": get_header_index(tie_label),
            f"TTIE0({i:.0f})": ttie0,
            # TODO: What if we don't want to tie it back to first INDV element?
            f"TTIE({i:.0f},{indv[0]:.0f})": ttie
        })
    
    # Freeze all other labels.
    frozen_parameters = { hln: (hln not in c["INDV_LABEL"]) for hln in header_label_names }

    return (frozen_parameters, ferre_kwds)