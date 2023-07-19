#!/usr/bin/env python3
import click

# Common options.
@click.command()
@click.argument("dir")
@click.option("-n", default=128, help="Number of threads", show_default=True)
@click.option("--memory", is_flag=True, default=True, help="Load the FERRE grid into memory", show_default=True)
def post_execution_interpolation(dir, n=128, memory=True, epsilon=0.001):
    """
    Run a single FERRE process that interpolates spectra at the best-fitting model parameters,
    without applying any normalization. 
    
    If the best-fitting parameters are on a grid edge then usually FERRE returns NaNs, so a
    small epsilon is applied.
    """
    f_access = 1 if memory else 0
    import os
    import numpy as np
    from astra.utils import expand_path
    from astra.pipelines.ferre.utils import (
        execute_ferre, parse_control_kwds, read_ferre_headers, format_ferre_input_parameters
    )

    input_path = expand_path(f"{dir}/input.nml")
    control_kwds = parse_control_kwds(input_path)

    # parse synthfile headers to get the edges
    synthfile = control_kwds["SYNTHFILE(1)"]
    headers = read_ferre_headers(synthfile)

    output_parameter_path = os.path.join(f"{dir}/{os.path.basename(control_kwds['OPFILE'])}")
    output_names = np.atleast_1d(np.loadtxt(output_parameter_path, usecols=(0, ), dtype=str))
    output_parameters = np.atleast_2d(np.loadtxt(output_parameter_path, usecols=range(1, 1 + int(control_kwds["NDIM"]))))

    clipped_parameters = np.clip(
        output_parameters, 
        headers[0]["LLIMITS"] + epsilon,
        headers[0]["ULIMITS"] - epsilon
    )

    clipped_parameter_path = f"{output_parameter_path}.clipped"
    with open(clipped_parameter_path, "w") as fp:
        for name, point in zip(output_names, clipped_parameters):
            fp.write(format_ferre_input_parameters(*point, name=name))

    contents = f"""
    &LISTA
    SYNTHFILE(1) = '{synthfile}'
    NOV = 0
    OFFILE = 'model_flux.output'
    PFILE = '{os.path.basename(clipped_parameter_path)}'
    INTER = {control_kwds['INTER']}
    F_FORMAT = 1
    F_ACCESS = {f_access}
    F_SORT = 0
    NTHREADS = {n}
    NDIM = {control_kwds['NDIM']}
    /
    """
    contents = "\n".join(map(str.strip, contents.split("\n"))).lstrip()
    input_nml_path = f"{dir}/post_execution_interpolation.nml"
    with open(input_nml_path, "w") as fp:
        fp.write(contents)
    
    execute_ferre(input_nml_path)
    print(f"Wrote un-normalized model spectra to {os.path.join(dir, 'model_flux.output')}")
    return None


if __name__ == "__main__":
    post_execution_interpolation()
