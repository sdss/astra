"""General FERRE utilities."""

import os
import datetime
import numpy as np
import re
import subprocess
from typing import Optional
from tqdm import tqdm
from glob import glob
from itertools import cycle
from astra.utils import log, expand_path


TRANSLATE_LABELS = { 
    "teff": "TEFF",
    "logg": "LOGG",
    "log10_v_micro": "LOG10VDOP",
    "log10_v_sini": "LGVSINI",
    "m_h": "METALS",
    "alpha_m": "O Mg Si S Ca Ti",
    "c_m": "C",
    "n_m": "N",
}

def get_input_spectrum_identifiers(stage_dir):
    """
    Get all spectrum identifiers analyzed in a given stage.
    
    :param stage_dir:
        The stage directory (e.g., ``/your/parent_folder/coarse``)
    """
    spectrum_ids = set()
    for path in glob(f"{expand_path(stage_dir)}/*/parameter.input"):
        for name in np.atleast_1d(np.loadtxt(path, usecols=(0, ), dtype=str)):
            spectrum_ids.add(parse_ferre_spectrum_name(name)["spectrum_id"])
    return spectrum_ids




def parse_control_kwds(input_nml_path):
    with open(expand_path(input_nml_path), "r") as fp:
        contents = fp.read()
    matches = re.findall("(?P<key>[A-Z\(\)0-9]+)\s*=\s*['\"]?\s*(?P<value>.+)\s*['\"]?\s*$", contents, re.MULTILINE)
    control_kwds = {}
    for key, value in matches:
        value = value.rstrip('"\'')
        try:
            value = int(value)
        except:
            None
        finally:
            control_kwds[key] = value
    return control_kwds

def execute_ferre(path, timeout=None):
        
    path = expand_path(path)
    if path.lower().endswith(".nml"):
        # A direct NML file was given.
        input_nml_path = os.path.basename(path)
        pwd = os.path.dirname(path)
        stdout_path = os.path.join(pwd, f"{input_nml_path}.stdout")
        stderr_path = os.path.join(pwd, f"{input_nml_path}.stderr")

    else:
        pwd = path
        input_nml_path = "input.nml"
        stdout_path = os.path.join(pwd, "stdout")
        stderr_path = os.path.join(pwd, "stderr")

    log.debug(f"Running FERRE on {pwd}/{input_nml_path} with timeout of {timeout}")
    try:
        with open(stdout_path, "w") as stdout:
            with open(stderr_path, "w") as stderr:
                process = subprocess.run(
                    ["ferre.x", input_nml_path],
                    cwd=pwd,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                    timeout=timeout, 
                )
    except subprocess.TimeoutExpired:
        log.exception(f"FERRE has timed out in {pwd}")
    except:
        log.exception(f"Exception when calling FERRE in {pwd}:")
        raise

    else:
        log.debug(f"Ferre finished")

    return None


def get_apogee_segment_indices():
    # TODO: put elsewhere
    segment_log10_wl_start = np.array([4.180476e+00, 4.20051e+00, 4.217064e+00])
    segment_pixels = np.array([3028, 2495, 1991])
    start_indices = np.round((segment_log10_wl_start - 4.179) / 6e-6).astype(int)
    return (start_indices, segment_pixels)

def get_apogee_pixel_mask():
    # TODO: put elsewhere?
    mask = np.zeros(8575, dtype=bool)
    for si, p in zip(*get_apogee_segment_indices()):
        mask[si:si+p] = True
    assert mask.sum() == 7514
    return mask

'''
def mask_apogee_pixel_array(pixel_array):
    pixel_array = np.atleast_2d(pixel_array)
    N, P = pixel_array.shape
    assert P == 8575, "Expected full array (8575 pixels) to mask"
    mask = get_apogee_pixel_mask()

    return pixel_array[mask]


def de_mask_apogee_pixel_array(ferre_array, fill_value=np.nan):
    ferre_array = np.atleast_2d(ferre_array)
    N, P = ferre_array.shape
    assert P == 7514, "Expected FERRE array (7514 pixels) to mask"

    start_indices, segment_pixels = get_apogee_segment_indices()
    pixel_array = fill_value * np.ones((N, 8575))

    si = 0
    for start_index, pixels in zip(start_indices, segment_pixels):
        pixel_array[start_index:start_index + pixels] = ferre_array[si:si + pixels]
        si += pixels
    assert np.sum(pixel_array != fill_value) == 7514
    return pixel_array
'''              





def get_ferre_spectrum_name(*args):
    return "_".join(map(str, args))
    
def int_or_none(_):
    try:
        return int(_)
    except:
        return None

def parse_ferre_spectrum_name(name):
    index, source_id, spectrum_id, initial_flags, upstream_id = map(int_or_none, name.split("_"))

    return dict(
        index=index,
        source_id=source_id,
        spectrum_id=spectrum_id,
        initial_flags=initial_flags,
        upstream_id=upstream_id
    )


def get_ferre_label_name(parameter_name, ferre_label_names, transforms=None):
    transforms = transforms or TRANSLATE_LABELS

    try:
        return transforms[parameter_name]
    except:
        if parameter_name in ferre_label_names:
            return parameter_name
        
    raise ValueError(f"Cannot match {parameter_name} among {ferre_label_names}")



def parse_header_path(header_path):
    """
    Parse the path of a header file and return a dictionary of relevant parameters.

    :param header_path:
        The path of a grid header file.

    :returns:
        A dictionary of keywords that are relevant to running FERRE tasks.
    """

    (
        *_,
        radiative_transfer_code,
        model_photospheres,
        isotopes,
        folder,
        basename,
    ) = header_path.split("/")

    parts = basename.split("_")
    # p_apst{gd}{spectral_type}_{date}_lsf{lsf}_{aspcap}_012_075
    _ = 4
    gd, spectral_type = (parts[1][_], parts[1][_ + 1 :])
    # Special case for the BA grid with kurucz atmospheres. Sigh.
    if gd == "B" and spectral_type == "A":
        year, month, day = (2019, 11, 21)
        lsf = "combo5"
        lsf_telescope_model = "lco25m" if parts[2].endswith("s") else "apo25m"
        is_giant_grid = False
        gd = ""
        spectral_type = "BA"

    else:
        date_str = parts[2]
        year, month, day = (
            2000 + int(date_str[:2]),
            int(date_str[2:4]),
            int(date_str[4:6]),
        )
        if radiative_transfer_code == "turbospec":        
            lsf = parts[-3].rstrip("s")[-1]
            lsf_telescope_model = "lco25m" if parts[-3].endswith("s") else "apo25m"
        else:
            lsf = parts[3][3]
            lsf_telescope_model = "lco25m" if parts[3][4:] == "s" else "apo25m"

        is_giant_grid = gd == "g"

    short_grid_name = f"{spectral_type}{gd}_{lsf}"

    kwds = dict(
        radiative_transfer_code=radiative_transfer_code,
        model_photospheres=model_photospheres,
        isotopes=isotopes,
        gd=gd,
        lsf_telescope_model=lsf_telescope_model,
        spectral_type=spectral_type,
        grid_creation_date=datetime.date(year, month, day),
        lsf=lsf,
        short_grid_name=short_grid_name
    )

    return kwds




def validate_ferre_control_keywords(
    header_path,
    frozen_parameters=None,
    interpolation_order=3,
    weight_path=None,
    lsf_shape_path=None,
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
    n_runs=1,
    f_access=None,
    f_format=1,
    f_sort=False,
    **kwargs,
):
    """
    Validate the FERRE control keywords.

    :param header_path:
        The path of the FERRE header file.

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

    :param weight_path: [optional]
        The location of a weight (or mask) file to apply to the pixels. This corresponds
        to the FERRE keyword `filterfile`.

    :para lsf_shape_path: [optional]
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
           `continuum_order` and `continuum_reject` keywords). Note that even if this
           option is selected, FERRE will perform continuum normalization on a per-segment
           level, where the segment here refers to the model segments stored in the 
           FERRE header file (e.g., one segment per chip). You can confirm this by
           setting `continuum_flag=1`, `continuum_order=0` and you will see that
           each chip has been continuum normalised separately.
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
    
    :param f_sort: [optional]
        Ask FERRE to sort the outputs to be the same ordering as the inputs (default: False)
        
        WARNING: FERRE does this in a very inefficient way. The sorting is an N^2
        operation that is performed ON DISK (i.e., it is not done in memory). This
        means it can take a huge time just to sort the outputs after the main
        execution is complete. It's recommended that you let Astra do this for
        you post-execution.
    """

    header_path = expand_path(header_path)
    kwds = {
        "synthfile(1)": header_path,
        "pfile": "parameter.input",
        "wfile": "wavelength.input",
        "ffile": "flux.input",
        "erfile": "e_flux.input",
        "opfile": "parameter.output",
        # This is to distinguish it from when we do the interpolation run and get the (unaltered) model flux
        "offile": "rectified_model_flux.output", 
        "sffile": "rectified_flux.output",
    }
    headers, *segment_headers = read_ferre_headers(header_path)

    # Parse frozen parameters.
    ferre_label_names = headers["LABEL"]

    frozen_parameters = frozen_parameters or dict()
    if frozen_parameters:
        frozen_label_names = {}
        for parameter_name, state in frozen_parameters.items():
            if isinstance(state, bool) and state:
                try:
                    ferre_label_name = get_ferre_label_name(parameter_name, ferre_label_names)
                except:
                    log.warning(f"Ignoring unknown parameters given in frozen parameters: {parameter_name} (available: {ferre_label_names})")

                else:
                    frozen_label_names[ferre_label_name] = state

        indices = [
            i
            for i, label_name in enumerate(ferre_label_names, start=1)
            if label_name not in frozen_label_names
        ]
        if len(indices) == 0:
            raise ValueError(f"All parameters frozen?!")

    else:
        # No frozen parameters
        indices = 1 + np.arange(len(ferre_label_names), dtype=int)

    L = len(indices)
    kwds.update(
        {
            "ndim": headers["N_OF_DIM"],
            "nov": L,
            "indv": " ".join([f"{i:.0f}" for i in indices]),
            "init": 0,
            "nruns": n_runs,
            #"indini": " ".join(["1"] * L),
            "inter": validate_interpolation_order(interpolation_order),
            "errbar": validate_error_algorithm_flag(error_algorithm_flag),
            "algor": validate_optimization_algorithm_flag(optimization_algorithm_flag),
            "pcachi": int(pca_chi),
            "pcaproject": int(pca_project),
            "covprint": int(full_covariance),
            "nthreads": int(n_threads),
            "f_access": int(f_access or False),
            "f_format": int(f_format),
            "f_sort": int(f_sort),
        }
    )
    wavelength_interpolation_flag = validate_wavelength_interpolation_flag(
        wavelength_interpolation_flag
    )
    if wavelength_interpolation_flag > 0:
        kwds.update({"winter": wavelength_interpolation_flag})

    lsf_shape_flag, lsf_shape_path = validate_lsf_shape_flag_and_lsf_shape_path(
        lsf_shape_flag, lsf_shape_path
    )
    if lsf_shape_flag is not None:
        kwds.update(
            {
                "lsf": lsf_shape_flag,
                "lsffile": lsf_shape_path,
            }
        )
    if weight_path is not None:
        kwds["filterfile"] = validate_weight_path(weight_path)

    # Continuum args.
    kwds.update(
        validate_continuum_arguments(
            continuum_flag=continuum_flag,
            continuum_order=continuum_order,
            continuum_segment=continuum_segment,
            continuum_reject=continuum_reject,
            continuum_observations_flag=continuum_observations_flag,
        )
    )

    kwds.update(kwargs or dict())
    return (kwds, headers, segment_headers, frozen_parameters)


def format_ferre_control_keywords(ferre_kwds: dict) -> str:
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
        #"indini",
        "winter",
        "algor",
        "lsf",
        "nthreads",
        "covprint",
        "pcachi",
        "f_format",
        "f_access",
        "f_sort",
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


def sanitise(parameter_name):
    if isinstance(parameter_name, (list, tuple, set, np.ndarray)):
        return list(map(sanitise, parameter_name))
    return parameter_name.lower().strip().replace(" ", "_")


def validate_interpolation_order(interpolation_order) -> int:
    available = {
        0: "nearest neighbour",
        1: "linear",
        2: "quadratic Bezier",
        3: "cubic Bezier",
        4: "cubic splines",
    }
    interpolation_order = int(interpolation_order)
    if interpolation_order not in available:
        raise ValueError(
            f"interpolation_order must be one of {tuple(list(available.keys()))}"
        )
    return interpolation_order


def validate_weight_path(weight_path) -> Optional[str]:
    if weight_path is not None:
        weight_path = expand_path(weight_path)
        if not os.path.exists(weight_path):
            raise ValueError(f"the weight_path does not exist: {weight_path}")
        return weight_path


def validate_lsf_shape_flag_and_lsf_shape_path(lsf_shape_flag, lsf_shape_path):
    available = (0, 1, 2, 3, 4, 11, 12, 13, 14)
    if lsf_shape_flag is not None:
        lsf_shape_flag = int(lsf_shape_flag)
        if lsf_shape_flag not in available:
            raise ValueError(f"lsf_shape_flag must be one of {available}")

        if lsf_shape_flag != 0:
            if lsf_shape_path is None:
                raise ValueError(
                    f"lsf_shape_flag is not 0, so an `lsf_shape_path` is needed"
                )

            lsf_shape_path = expand_path(lsf_shape_path)
            if not os.path.exists(lsf_shape_path):
                raise ValueError(f"lsf_shape_path does not exist: {lsf_shape_path}")
            return (lsf_shape_flag, lsf_shape_path)
    return (None, None)


def validate_error_algorithm_flag(error_algorithm_flag):
    available = {
        0: "adopt distance from the solution at which \chi^2 = min(\chi^2) + 1",
        1: "invert the curvature matrix",
        2: "perform numerical experiments injecting noise into the data",
    }
    error_algorithm_flag = int(error_algorithm_flag)
    if error_algorithm_flag not in available:
        raise ValueError(
            f"error_algorithm_flag must be one of {tuple(list(available.keys()))}"
        )
    return error_algorithm_flag


def validate_wavelength_interpolation_flag(wavelength_interpolation_flag):
    available = (0, 1, 2)
    wavelength_interpolation_flag = int(wavelength_interpolation_flag or 0)
    if wavelength_interpolation_flag not in available:
        raise ValueError(f"wavelength_interpolation_flag must be one of {available}")
    # if wavelength_interpolation_flag > 0:
    #    if wavelength is None:
    #        raise ValueError("if wavelength_interpolation_flag != 0 then wavelength must be given")
    #    return dict(winter=wavelength_interpolation_flag)
    return wavelength_interpolation_flag


def validate_optimization_algorithm_flag(optimization_algorithm_flag):
    available = (1, 2, 3, 4)
    optimization_algorithm_flag = int(optimization_algorithm_flag)
    if optimization_algorithm_flag not in available:
        raise ValueError(f"optimization_algorithm_flag must be one of {available}")
    return optimization_algorithm_flag


def validate_continuum_arguments(
    continuum_flag,
    continuum_order,
    continuum_segment,
    continuum_reject,
    continuum_observations_flag,
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
                ncont=int(continuum_order),
            )

        elif continuum_flag in (2, 3):
            if continuum_segment is None:
                raise ValueError(
                    "continuum_segment is required if continuum_flag is 2 or 3"
                )
            if continuum_flag == 2:
                continuum_segment = int(continuum_segment)
            elif continuum_flag == 3:
                continuum_segment = float(continuum_segment)
            if continuum_segment < 1:
                raise ValueError(f"continuum_segment must be a positive value")

            kwds.update(cont=continuum_flag, ncont=continuum_segment)

    if continuum_observations_flag is not None:
        kwds.update(obscont=int(continuum_observations_flag))

    return kwds


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
                values = re.findall("[+|-|\d|\.|e|-]+", value)

                dtype = float if "." in value else int

                if len(values) == 1:
                    value = dtype(values[0])
                else:
                    value = np.array(values, dtype=dtype)

            else:
                value = value.strip(" '\n")

            match = re.match(r"\s+(?P<key>\w+)\((?P<index>\d+)\)", key)
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
    header["ULIMITS"] = header["LLIMITS"] + header["STEPS"] * (header["N_P"] - 1)

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


def validate_initial_and_frozen_parameters(
    headers,
    initial_parameters,
    frozen_parameters,
    transforms=None,
    clip_initial_parameters_to_boundary_edges=True,
    clip_epsilon_percent=1,
):

    N = len(initial_parameters)
    
    ferre_label_names = headers["LABEL"]

    mid_point = grid_mid_point(headers)
    initial_parameters_array = np.tile(mid_point, N).reshape((N, -1))
    warning_messages = []

    for i, ip in enumerate(initial_parameters):
        for parameter_name, value in ip.items():

            try:
                ferre_label_name = get_ferre_label_name(parameter_name, ferre_label_names, transforms)
            except:
                message = f"Ignoring initial parameter '{parameter_name}' as it is not in {ferre_label_names}"
                if message not in warning_messages:
                    log.warning(message)
                    warning_messages.append(message)
                continue
        
            else:
                if ferre_label_name not in ferre_label_names:
                    message = f"Ignoring initial parameter '{ferre_label_name}' as it is not in {ferre_label_names}"
                    if message not in warning_messages:
                        log.warning(message)
                        warning_messages.append(message)
                    continue

            if np.isfinite(value):
                j = ferre_label_names.index(ferre_label_name)
                initial_parameters_array[i, j] = value

    # Update with frozen parameters
    for parameter_name, value in frozen_parameters.items():
        if not isinstance(value, bool):
            try:
                ferre_label_name = get_ferre_label_name(parameter_name, ferre_label_names, transforms)
            except:
                message = f"Ignoring frozen parameter '{parameter_name}' as it is not in {ferre_label_names}"
                if message not in warning_messages:
                    log.warning(message)
                    warning_messages.append(message)
                continue

            j = ferre_label_names.index(ferre_label_name)
            log.info(f"Over-writing initial values for {parameter_name} ({ferre_label_name}) with frozen value of {value}")
            initial_parameters_array[:, j] = value


    # Let's check the initial values are all within the grid boundaries.
    # TODO: These should already be clipped. If we are hitting  this then it's a problem.
    lower_limit, upper_limit = grid_limits(headers)
    try:
        check_initial_parameters_within_grid_limits(
            initial_parameters_array, lower_limit, upper_limit, headers['LABEL']
        )
    except ValueError as e:
        log.exception(
            f"Exception when checking initial parameters within grid boundaries:"
        )
        log.critical(e, exc_info=True)

        if clip_initial_parameters_to_boundary_edges:
            log.info(
                f"Clipping initial parameters to boundary edges (use clip_initial_parameters_to_boundary_edges=False to raise exception instead)"
            )

            clip = clip_epsilon_percent * (upper_limit - lower_limit) / 100.0
            initial_parameters_array = np.round(
                np.clip(
                    initial_parameters_array, lower_limit + clip, upper_limit - clip
                ),
                3,
            )
        else:
            raise

    return initial_parameters_array



def clip_initial_guess(initial_guess, headers, transforms=None, percent_epsilon=1, decimals=2):
    """
    Clip the input initial guess to some epsilon within the bounds of the FERRE grid.

    :param initial_guess:
        A dictionary containing the input initial guess, with parameter names (arbitrary) as keys
        and values as the initial guess for that parameter.
    
    :param headers:
        The FERRE grid headers.
    
    :param transforms: [optional]
        A dictionary containing parameter names (as per `initial_guess`) as keys, and the actual
        FERRE grid label that corresponds to as the values. If `None` is given then `TRANSLATE_LABELS`
        will be used.
    
    :param percent_epsilon: [optional]
        The percentage of the grid step size to clip the initial guess by.
    
    :param decimals: [optional]
        The number of decimal places to round the clipped initial guess to.
    """
    clip = percent_epsilon * (headers["ULIMITS"] - headers["LLIMITS"]) / 100.0
    lower_limits, upper_limits = (headers["LLIMITS"] + clip, headers["ULIMITS"] - clip)

    if decimals is not None:
        lower_limits = np.round(lower_limits, decimals)
        upper_limits = np.round(upper_limits, decimals)
    
    clipped_initial_guess = {}
    transforms = transforms or TRANSLATE_LABELS
    for parameter, value in initial_guess.items():

        try:
            label_name = transforms[parameter]
        except KeyError:
            if parameter not in headers["LABEL"]:
                # No clipping to apply!
                clipped_initial_guess[parameter] = value
                continue
        else:

            try:
                index = headers["LABEL"].index(label_name)
            except ValueError:
                # No clipping to apply!
                clipped_initial_guess[parameter] = value
                continue

            else:    
                clipped_value = np.clip(value, lower_limits[index], upper_limits[index])

                if decimals is not None:
                    clipped_value = np.round(clipped_value, decimals)
                clipped_initial_guess[parameter] = clipped_value
    return clipped_initial_guess
            


def wavelength_array(header):
    wave = header["WAVE"][0] + np.arange(header["NPIX"]) * header["WAVE"][1]
    if header["LOGW"] == 1:
        wave = 10**wave
    elif header["LOGW"] == 2:
        wave = np.exp(wave)
    return wave


def format_ferre_input_parameters(*p, name="dummy"):
    r"""
    Format input parameters for FERRE to digest.

    :returns:
        String contents for a FERRE input parameter file.
    """
    contents = f"{name:40s}"
    for each in p:
        contents += f"{each:12.3f}"
    contents += "\n"
    return contents


def grid_mid_point(headers):
    return np.mean(np.vstack([headers["LLIMITS"], headers["ULIMITS"]]), axis=0)


def grid_limits(headers):
    """
    Return a two-length tuple that contains the lower limits of the grid and the upper limits of the grid.

    :param headers:
        The primary headers from a FERRE grid file.
    """
    return (
        headers["LLIMITS"],
        headers["LLIMITS"] + headers["STEPS"] * (headers["N_P"] - 1),
    )


def check_initial_parameters_within_grid_limits(
    initial_parameters, lower_limit, upper_limit, parameter_names
):
    """
    Check that the initial parameters are within the boundaries of the FERRE grid.

    :param initial_parameters:
        A 2D array of shape (N, L) where N is the number of spectra and L is the number of labels in the FERRE grid.

    :param headers:
        The primary headers from the FERRE grid file.

    :raise ValueError:
        If any initial parameter is outside the grid boundary.
    """

    in_limits = (upper_limit > initial_parameters) * (initial_parameters > lower_limit)

    if not np.all(in_limits):

        message = (
            "Initial_parameters are not all within bounds of the grid. For example:\n"
        )

        bad_parameter_indices = np.where(~np.all(in_limits, axis=0))[0]
        for j in bad_parameter_indices:
            i = np.where(~in_limits[:, j])[0]
            message += f"- {parameter_names[j]} has limits of ({lower_limit[j]:.2f}, {upper_limit[j]:.2f}) but {len(i)} indices ({i}) are outside this range: {initial_parameters[:, j][i]}\n"

        raise ValueError(message)

    return True


def read_control_file(path):
    routes = [
        ("FFILE", "FFILE"),
        ("ERFILE", "ERFILE"),
        ("OFFILE", "OFFILE"),
        ("OPFILE", "OPFILE"),
        ("SFFILE", "SFFILE"),
        ("PFILE", "PFILE"),
        ("FILTERFILE", "FILTERFILE"),
        ("NDIM", "NDIM"),
        ("INDV", "INDV"),
        ("F_FORMAT", "F_FORMAT"),
        ("F_ACCESS", "F_ACCESS"),
        ("NCONT", "NCONT"),
        ("INTER", "INTER"),
        ("REJECTCONT", "REJECTCONT"),
        ("NTHREADS", "NTHREADS"),
        ("OBSCONT", "OBSCONT"),
        ("COVPRINT", "COVPRINT"),
        ("SYNTHFILE(1)", "SYNTHFILE(1)"),
    ]

    meta = {}
    with open(path, "r") as fp:
        for line in fp.readlines():
            for route, key in routes:
                if line.startswith(route):
                    meta[key] = line.split("=")[1].strip(" \n'")

    return meta


def wc(path):
    return int(subprocess.check_output(["wc", "-l", path]).split()[0])


def parse_ferre_output(dir, stdout, stderr, control_file_basename="input.nml"):
    control_kwds = read_control_file(os.path.join(dir, control_file_basename))
    input_path = control_kwds["PFILE"]
    output_path = control_kwds["OFFILE"]

    total = wc(os.path.join(dir, input_path))
    n_done = wc(os.path.join(dir, output_path))
    n_errors = stderr.lower().count("error")

    meta = {}
    for line in stdout.split("\n"):
        if "f e r r e" in line:
            meta["ferre_version"] = line.strip().split()[-1]
    

    return (n_done, n_errors, control_kwds, meta)


def check_ferre_progress(
    dir, process=None, control_file_basename="input.nml", timeout=30
):

    control_kwds = read_control_file(os.path.join(dir, control_file_basename))
    input_path = control_kwds["PFILE"]
    output_path = control_kwds["OFFILE"]

    total = wc(os.path.join(dir, input_path))

    stdout, stderr = ("", "")

    total_done, total_errors = (0, 0)
    with tqdm(total=total, desc="FERRE", unit="spectra") as pb:
        while total > total_done:
            if process is not None:
                try:
                    _stdout, _stderr = process.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    None
                else:
                    stdout += _stdout
                    stderr += _stderr

            n_done = wc(os.path.join(dir, output_path))

            n_errors = stderr.lower().count("error")

            n_updated = n_done - total_done
            pb.update(n_updated)

            total_done = n_done
            total_errors = n_errors

            if n_errors > 0:
                pb.set_description(f"FERRE ({total_errors:.0f} errors)")
            pb.refresh()

    if process is not None:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            None
        else:
            stdout += stdout
            stderr += stderr
            total_errors = stderr.lower().count("error")

    return (stdout, stderr, total_done, total_errors, control_kwds)


def read_and_sort_output_data_file(path, input_names, n_data_columns=None, dtype=float):
    names = np.atleast_1d(np.loadtxt(path, usecols=(0, ), dtype=str))
    if n_data_columns is None:
        with open(path, "r") as fp:
            n_data_columns = len(fp.readline().strip().split()) - 1

    data = np.atleast_2d(np.loadtxt(path, usecols=range(1, 1 + n_data_columns), dtype=dtype))
    return sort_data_as_per_input_names(input_names, names, data)


def read_file_with_name_and_data(path, input_names, n_data_columns=None, dtype=float):
    names = np.atleast_1d(np.loadtxt(path, usecols=(0, ), dtype=str))
    # Need the number of columns.
    if n_data_columns is None:
        with open(path, "r") as fp:
            n_data_columns = len(fp.readline().strip().split()) - 1
    
    data = np.atleast_2d(np.loadtxt(path, usecols=range(1, 1 + n_data_columns), dtype=dtype))
    if input_names is None:
        return (names, data)

    data = sort_data_as_per_input_names(input_names, names, data)
    raise a # "check above"
    missing_names = set(input_names).difference(names)
    return (data, missing_names)

def read_input_parameter_file(pwd, control_kwds):
    return read_file_with_name_and_data(os.path.join(pwd, control_kwds["PFILE"]), None)

def read_input_data_file(path):
    return np.atleast_2d(np.loadtxt(path))





def read_output_parameter_file(pwd, control_kwds, input_names):

    output_parameters_path = os.path.join(pwd, control_kwds["OPFILE"])

    n_dimensions = int(control_kwds["NDIM"])
    full_covariance = bool(int(control_kwds["COVPRINT"]))

    return _read_output_parameter_file(output_parameters_path, n_dimensions, full_covariance, input_names)


def _read_output_parameter_file(path, n_dimensions, full_covariance, input_names):
    
    names = np.atleast_1d(np.loadtxt(path, usecols=(0,), dtype=str))

    N_cols = 2 * n_dimensions + 3
    if full_covariance:
        N_cols += n_dimensions**2

    results = np.atleast_2d(np.loadtxt(path, usecols=1 + np.arange(N_cols)))
    # sort here if we get given a set of input names
    results, missing_names, output_indices = sort_data_as_per_input_names(input_names, names, results)

    param = results[:, 0:n_dimensions]
    P, L = param.shape

    param_err = results[:, n_dimensions : 2 * n_dimensions]
    if full_covariance:
        cov = results[:, 2*n_dimensions + 3:].reshape((P, L, L))
    else:
        cov = cycle(np.eye(L))

    frac_phot_data_points, log_snr_sq, log_chisq_fit = results[
        :, 2 * n_dimensions : 2 * n_dimensions + 3
    ].T

    meta = dict(
        frac_phot_data_points=frac_phot_data_points,
        log_snr_sq=log_snr_sq,
        log_chisq_fit=log_chisq_fit,
        cov=cov,
    )

    return (param, param_err, meta, missing_names)




def sort_data_as_per_input_names(input_names, unsorted_names, unsorted_data):
    _, D = unsorted_data.shape
    N = len(input_names)

    data = np.nan * np.ones((N, D), dtype=float)
    intersect, input_index, output_index = np.intersect1d(
        input_names, 
        unsorted_names,
        assume_unique=False, 
        return_indices=True
    )
    data[input_index] = unsorted_data[output_index]
    missing = set(input_names) - set(intersect)
    output_indices = -1 * np.ones(N)
    output_indices[input_index] = output_index
    return (data, missing, output_indices)



def get_processing_times(stdout):
    """
    Get the time taken to analyse spectra and estimate the initial load time.

    :param stdout: (optional)
        The standard output from FERRE.
    """

    if stdout is None or stdout == "":
        return None

    headers = re.finditer("-{65}\s+f e r r e", stdout)
    start = next(headers)
    try:
        end = next(headers)
    except:
        None
    else:
        stdout = stdout[start.span()[1]:end.span()[0]]
    

    #i = stdout[::-1].index(header[0][::-1])
    #use_stdout = stdout[-(i + len(header)) :]

    n_threads = int(re.findall("nthreads = \s+[0-9]+", stdout)[0].split()[-1])
    n_obj = int(re.findall("nobj = \s+[0-9]+", stdout)[0].split()[-1])

    n_per_thread = n_obj // n_threads
    n_mod = n_obj - n_per_thread * n_threads

    si = 0
    expected_indices = -1 * np.ones((n_threads, n_per_thread + 1), dtype=int)
    for i in range(n_threads):
        ei = n_per_thread + (1 if n_mod > i else 0)
        expected_indices[i, :ei] = range(si, si + ei)
        si += ei        

    # Find the obvious examples first.
    elapsed_time_pattern = "ellapsed time:\s+(?P<time>[{0-9}|.]+)\s*s?\s*"
    next_object_pattern = "next object #\s+(?P<index_plus_one>[0-9]+)"

    elapsed_times = np.nan * np.ones(expected_indices.shape)

    matcher = re.finditer(elapsed_time_pattern, stdout)
    time_load, = next(matcher).groups()
    elapsed_times[:, 0] = float(time_load)

    for match in matcher:
        elapsed_time, = match.groups()
        si, ei = match.span()
        try:
            oi = stdout[ei:].index("\n")
        except ValueError:
            continue
        next_object_match = re.match(next_object_pattern, stdout[ei:ei+oi])
        if next_object_match:
            next_object, = next_object_match.groups()
            i, j = np.where(expected_indices == (int(next_object) - 1 - 1)) # -1 for zero indexing, -1 to reference the object that was analysed
            elapsed_times[i, j] = float(elapsed_time)
        else:
            log.warning(f"Unassigned elapsed time: {elapsed_time}")

            raise a

    '''
    
In [74]: !grep None stdout | head -n 30 | grep None | awk '{print $1}' | sort
1
104
109
114
119
124
129
13
139
144
149
154
159
164
19
25
31
37
43
49
59
64
69
7
74
79
84
89
94
99

In [75]: !grep next stdout | head -n 30 | awk '{print $4}' | sort
1
104
109
114
119
124
129
13
134
139
144
149
154
159
164
19
25
31
37
43
49
54
59
64
69
7
74
79
89
94

In [76]: !pwd
/scratch/general/nfs1/u6020307/pbs/20230716_ipl3_1.1/ASPCAP-100/coarse/GKd_b

    '''

    raise a


    # There are many ways to match up the elapsed time per object. Some are more explicit
    # than others, but to require a more explicit matching means always depending on less
    # implicit circumstances anyways.
    elapsed_time_per_spectrum = np.nan * np.ones(n_obj)
    for index, elapsed_time in zip(object_indices, time_elapsed_unordered):
        elapsed_time_per_spectrum[index] = elapsed_time

    time_per_spectrum = np.nan * np.ones(n_obj)
    idx = np.sort(object_indices[:n_threads])
    for si, ei in np.hstack([0, np.repeat(idx[1:], 2), n_obj]).reshape((-1, 2)):
        time_per_spectrum[si:ei] = np.diff(
            np.hstack([time_load, elapsed_time_per_spectrum[si:ei]])
        )

    L, M = (len(time_elapsed_unordered), len(object_indices))
    if M < n_obj:
        log.warning(
            f"Could not find all object indices from FERRE stdout: expected {n_obj} found {M}"
        )
    if L < n_obj:
        log.warning(
            f"Could not find all elapsed times from FERRE stdout: expected {n_obj} found {L}"
        )

    return dict(
        ferre_time_load_grid=time_load,
        ferre_time_elapsed=time_per_spectrum,
        object_indices=object_indices,
        time_elapsed_unordered=time_elapsed_unordered,
        n_threads=n_threads,
        n_obj=n_obj,
    )
