
import datetime
import os
import numpy as np
import re
import subprocess

from collections import OrderedDict
from inspect import getfullargspec

from astra.utils import log

def sanitise_parameter_names(parameter_name):
    return parameter_name.lower().strip().replace(" ", "_")


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
            # TODO: switch to log
            log.exception(f"Unable to parse FERRE headers for {full_header_path}")
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
    # TODO: Check with Holtz on this microturbulence relation because last term is not used.
    DM = np.array([1, log_g, log_g**2, log_g**3, 0])
    return DM @ coeffs



def read_output_parameter_file(path, n_dimensions):
    
    """
    Three more columns follow, giving the fraction of photometric data
    points (useful when multiple grids combining spectroscopy and photometry are used),
    the average log(S/N)2
    for the spectrum, and the logarithm of the reduced χ^2 for the fit.

    Additional columns with the covariance matrix of the errors can be output setting to 1 the
    keyword COVPRINT.
    """

    names = np.loadtxt(path, usecols=(0, ), dtype=str)
    
    results = np.atleast_2d(np.loadtxt(path, usecols=1 + np.arange(2 * n_dimensions + 3)))

    param = results[:, 0:n_dimensions]
    param_err = results[:, n_dimensions:2*n_dimensions]

    meta = dict(
        frac_phot_data_points=results[:, -3],
        log_snr_sq=results[:, -2],
        log_chisq_fit=results[:, -1]
    )
    return (names, param, param_err, meta)





def format_ferre_control_keywords(ferre_kwds):
    r"""
    Format control keywords for FERRE to digest.

    :param ferre_kwds:
        A dictionary of FERRE-recognized keywords, in a suitable order.

    :returns:
        String contents for a FERRE input file.
    """
    contents = "&LISTA\n"
    for k, v in ferre_kwds.items():
        v = f"'{v}'" if (isinstance(v, str) and k not in ("indv", "indini")) else f"{v}"
        contents += f"{k.upper()} = {v}\n"
    contents += "/\n"
    return contents




def __format_ferre_input_parameters(*p, star_name="dummy"):
    r"""
    Format input parameters for FERRE to digest.

    :returns:
        String contents for a FERRE input parameter file.
    """
    contents = f"{star_name} "
    for each in p:
        contents += f"{each:12.3f} "
    contents += "\n"
    return contents



# Note: this old_format_ferre_input_parameters certainly will match the IPF files on the SDSS SAS,
#       but are giving me FERRE problems right now. Check format later on.
def format_ferre_input_parameters(*p, star_name="dummy"):
    r"""
    Format input parameters for FERRE to digest.

    :returns:
        String contents for a FERRE input parameter file.
    """
    contents = f"{star_name:40s}"
    for each in p:
        contents += f"{each:12.3f}"
    contents += "\n"
    return contents


# Dictionary to translate from our (more human-readable?) keyword arguments to FERRE parameters
_translate_keyword = OrderedDict([
    # Mandatory:
    ("n_dimensions", "ndim"),
    # Note here we parse fit_parameters into nov and indv, for deep and good reasons that are too
    # verbose to fit in a notebook's margin.
    ("n_parameters", "nov"),
    ("parameter_search_indices", "indv"),
    ("grid_header_path", "synthfile(1)"),
    ("input_parameter_path", "pfile"),
    ("input_flux_path", "ffile"),
    ("input_uncertainties_path", "erfile"),
    ("output_parameter_path", "opfile"),
    ("output_flux_path", "offile"),
    ("output_normalized_input_flux_path", "sffile"),
    ("input_weights_path", "filterfile"),

    # Optional:
    ("continuum_flag", "cont"),
    ("continuum_order", "ncont"),
    ("continuum_reject", "rejectcont"),
    ("continuum_observations_flag", "obscont"),
    ("n_spectra", "nobj"),
    ("n_pixels", "nlambda"),
    ("synthfile_format_flag", "f_format"),
    ("use_direct_access", "f_access"),
    ("interpolation_order", "inter"),
    ("error_algorithm_flag", "errbar"),
    ("n_runs", "nruns"),
    ("init_flag", "init"),
    ("init_algorithm_flag", "indini"),
    ("input_wavelength_path", "wfile"),
    ("input_lsf_path", "lsffile"),
    ("wavelength_interpolation_flag", "winter"),
    ("optimization_algorithm_flag", "algor"),
    ("lsf_shape_flag", "lsf"),
    ("n_threads", "nthreads"),

    ("full_covariance", "covprint"),
    ("pca_project", "pcaproject"),
    ("pca_chi", "pcachi")
])

def prepare_ferre_input_keywords(
        grid_header_path,
        n_parameters, 
        parameter_search_indices,
        n_dimensions, 
        input_parameter_path="parameters.input",
        input_wavelength_path="wavelengths.input", 
        input_flux_path="flux.input", 
        input_lsf_path="lsf.input",
        input_weights_path="weights.input",
        input_uncertainties_path="uncertainties.input", 
        output_parameter_path="parameters.output", 
        output_flux_path="flux.output",
        output_normalized_input_flux_path="normalized_flux.output",
        continuum_flag=None,
        continuum_order=None,
        continuum_reject=None,
        continuum_observations_flag=None,
        n_spectra=None, 
        n_pixels=None, 
        interpolation_order=3,
        n_runs=1,
        init_flag=1, 
        init_algorithm_flag=1, 
        optimization_algorithm_flag=1,
        error_algorithm_flag=0, 
        wavelength_interpolation_flag=0,
        lsf_shape_flag=0, 
        n_threads=1,
        synthfile_format_flag=1, 
        full_covariance=False,
        pca_project=False,
        pca_chi=False,
        use_direct_access=True,
        ferre_kwds=None
    ):

    arg_spec = getfullargspec(prepare_ferre_input_keywords)

    pca_chi = int(pca_chi)
    pca_project = int(pca_project)
    full_covariance = int(full_covariance)

    # Copy the arguments into one dictionary.
    kwds = OrderedDict([])
    for keyword in _translate_keyword.keys():
        if keyword in arg_spec.args:
            kwds[keyword] = eval(keyword)
        else:
            # It is expected that these keywords will be parsed and updated by this function.
            kwds[keyword] = None

    # Check interpolation order.
    available = {
        0: "nearest neighbour",
        1: "linear",
        2: "quadratic Bezier",
        3: "cubic Bezier",
        4: "cubic splines"
    }
    kwds["interpolation_order"] = int(kwds["interpolation_order"])
    if kwds["interpolation_order"] not in available.keys():
        raise ValueError(f"interpolation_order must be one of {available}")

    available = {
        0: "adopt distance from the solution at which \chi^2 = min(\chi^2) + 1",
        1: "invert the curvature matrix",
        2: "perform numerical experiments injecting noise into the data"
    }
    kwds["error_algorithm_flag"] = int(kwds["error_algorithm_flag"])
    if kwds["error_algorithm_flag"] not in available.keys():
        raise ValueError(f"error_algorithm_flag must be one of {available}")

    available = {
        0: "no interpolation", 
        1: "interpolate observations",
        2: "interpolate fluxes",
    }
    kwds["wavelength_interpolation_flag"] = int(kwds["wavelength_interpolation_flag"])
    if kwds["wavelength_interpolation_flag"] not in available.keys():
        raise ValueError(f"wavelength_interpolation_flag must be one of {available}")
    
    if kwds["wavelength_interpolation_flag"] == 0:
        # Won't need input wavelength path so remove it.
        kwds["input_wavelength_path"] = None
    
    #if kwds["wavelength_interpolation_flag"] == 2 and kwds["input_wavelength_path"] is None:
    #    raise ValueError("input_wavelength_path is needed if wavelength_interpolation_flag = 2")

    kwds["init_flag"] = int(kwds["init_flag"])
    if kwds["init_flag"] not in (0, 1):
        raise ValueError(f"init_flag must be one of (0, 1)")

    if isinstance(kwds["init_algorithm_flag"], int):
        v = np.ones(kwds["n_parameters"], dtype=int) * kwds["init_algorithm_flag"]
        kwds["init_algorithm_flag"] = " ".join([f"{_:.0f}" for _ in v])

    available = (0, 1, 2, 3, 4, 11, 12, 13, 14)
    kwds["lsf_shape_flag"] = int(kwds["lsf_shape_flag"])
    if kwds["lsf_shape_flag"] not in available:
        raise ValueError(f"lsf_shape_flag must be one of {available}")

    if kwds["synthfile_format_flag"] is None:
        # TODO: assuming only one synthfile_path
        _, ext = synthfile_path.rsplit(".", maxsplit=1)
        kwds["synthfile_format_flag"] = int(ext.lower() == "unf")

    kwds["synthfile_format_flag"] = int(kwds["synthfile_format_flag"])
    if kwds["synthfile_format_flag"] not in (0, 1):
        raise ValueError("synthfile_format_flag must be one of (0, 1)")

    kwds["use_direct_access"] = int(kwds["use_direct_access"])

    if continuum_flag is not None:
        available = (0, 1, 2)
        if continuum_flag not in available:
            raise ValueError(f"continuum_flag must be one of {available}")

        assert continuum_order is not None
    else:
        del kwds["continuum_flag"]
        del kwds["continuum_order"]
        del kwds["output_normalized_input_flux_path"]
        del kwds["continuum_reject"]

    remove_keywords_if_none = (
        "n_spectra",
        "n_pixels",
        "input_wavelength_path",
        "input_lsf_path",
        "input_flux_path", 
        "input_uncertainties_path", 
        "input_weights_path",
        "init_algorithm_flag",
        "continuum_observations_flag"
    )
    for keyword in remove_keywords_if_none:
        if kwds[keyword] is None:
            del kwds[keyword]

    if kwds["lsf_shape_flag"] == 0:
        try:
            del kwds["input_lsf_path"]
        except KeyError:
            None

    # Translate keywords.
    parsed_ferre_kwds = OrderedDict([])
    for k, v in kwds.items():
        if k == "parameter_search_indices":
            # Remember: indices are backwards for FERRE.
            #ferre_indices = np.sort(kwds["n_dimensions"] - v)
            parsed_ferre_kwds[_translate_keyword[k]] = " ".join(map(str, v))

        else:
            parsed_ferre_kwds[_translate_keyword[k]] = v

    if ferre_kwds is not None:
        parsed_ferre_kwds.update(ferre_kwds)

    return parsed_ferre_kwds