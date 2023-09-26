
import os
import numpy as np
from tempfile import TemporaryDirectory, mkdtemp
from astra.utils import expand_path
from astra.pipelines.ferre.utils import (
    execute_ferre, parse_control_kwds, read_ferre_headers, format_ferre_input_parameters,
    read_and_sort_output_data_file, get_apogee_pixel_mask
)
from shutil import rmtree


def interpolate(
    synthfile,
    teff,
    logg,
    m_h,
    alpha_m=None,
    c_m=None,
    n_m=None,
    log10_v_micro=None,
    log10_v_sini=None,
    inter=3,
    n_threads=32,
    read_in_memory=False,
    epsilon=1e-3,
    clip=True
):

    with TemporaryDirectory() as dir:
        
        f_access = 0 if read_in_memory else 1

        headers, *segment_headers = read_ferre_headers(synthfile)

        label_names = headers["LABEL"]

        translate = {
            "LOG10VDOP": log10_v_micro,
            "LGVSINI": log10_v_sini,
            "N": n_m,
            "C": c_m,
            "METALS": m_h,
            "O Mg Si S Ca Ti": alpha_m,
            "LOGG": logg,
            "TEFF": teff
        }

        parameters = np.atleast_2d([translate.get(ln) for ln in label_names]).T
        if clip:
            parameters = np.clip(
                parameters,
                headers["LLIMITS"] + epsilon,
                headers["ULIMITS"] - epsilon
            )
        else:
            raise a
            
        N = len(parameters)
        names = list(map(str, range(N)))
        input_parameter_path = f"parameter.input"
        output_model_flux_path = "model_flux.output"

        with open(os.path.join(dir, input_parameter_path), "w") as fp:
            for name, point in zip(names, parameters):
                fp.write(format_ferre_input_parameters(*point, name=name))

        contents = f"""
        &LISTA
        SYNTHFILE(1) = '{synthfile}'
        NOV = 0
        OFFILE = '{output_model_flux_path}'
        PFILE = '{input_parameter_path}'
        INTER = {inter}
        F_FORMAT = 1
        F_ACCESS = {f_access}
        F_SORT = 0
        NTHREADS = {n_threads}
        NDIM = {headers['N_OF_DIM']}
        /
        """
        contents = "\n".join(map(str.strip, contents.split("\n"))).lstrip()
        input_nml_path = f"{dir}/input.nml"
        with open(input_nml_path, "w") as fp:
            fp.write(contents)
        
        execute_ferre(input_nml_path)

        masked_model_flux, *_ = read_and_sort_output_data_file(os.path.join(dir, output_model_flux_path), names)
    
    mask = get_apogee_pixel_mask()  
    model_flux = np.nan * np.ones((N, 8575))
    model_flux[:, mask] = masked_model_flux

    return model_flux


def _pre_interpolate(
    synthfile,
    points,
    inter=3,
    n_threads=32,
    read_in_memory=False,
    epsilon=1e-3
):

    dir = mkdtemp()
    f_access = 0 if read_in_memory else 1

    headers, *segment_headers = read_ferre_headers(synthfile)

    points = np.clip(
        np.atleast_2d(points),
        headers["LLIMITS"] + epsilon,
        headers["ULIMITS"] - epsilon
    )

    N = len(points)
    names = list(map(str, range(N)))
    input_parameter_path = f"parameter.input"
    output_model_flux_path = "model_flux.output"

    with open(os.path.join(dir, input_parameter_path), "w") as fp:
        for name, point in zip(names, points):
            fp.write(format_ferre_input_parameters(*point, name=name))

    contents = f"""
    &LISTA
    SYNTHFILE(1) = '{synthfile}'
    NOV = 0
    OFFILE = '{output_model_flux_path}'
    PFILE = '{input_parameter_path}'
    INTER = {inter}
    F_FORMAT = 1
    F_ACCESS = {f_access}
    F_SORT = 0
    NTHREADS = {n_threads}
    NDIM = {headers['N_OF_DIM']}
    /
    """
    contents = "\n".join(map(str.strip, contents.split("\n"))).lstrip()
    input_nml_path = f"{dir}/input.nml"
    with open(input_nml_path, "w") as fp:
        fp.write(contents)    

    return input_nml_path


def _post_interpolate(input_nml_path, remove_dir=False):

    dir = os.path.dirname(input_nml_path)
    names = np.loadtxt(f"{dir}/parameter.input", usecols=(0, ), dtype=str)
    try:
        masked_model_flux, *_ = read_and_sort_output_data_file(os.path.join(dir, "model_flux.output"), names)
    except:
        print(f"Error in {dir}")
        raise
    else:
        N = len(names)
        mask = get_apogee_pixel_mask()  
        model_flux = np.nan * np.ones((N, 8575))
        model_flux[:, mask] = masked_model_flux
        if remove_dir:
            rmtree(dir)    

    return model_flux


def _interpolate(
    synthfile,
    points,
    inter=3,
    n_threads=32,
    read_in_memory=False,
    epsilon=1e-3
):

    input_nml_path = _pre_interpolate(
        synthfile,
        points,
        inter=inter,
        n_threads=n_threads,
        read_in_memory=read_in_memory,
        epsilon=epsilon
    )
    
    execute_ferre(input_nml_path)

    return _post_interpolate(input_nml_path, remove_dir=True)
