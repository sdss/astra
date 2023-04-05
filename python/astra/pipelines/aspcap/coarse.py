import numpy as np
import os
from astra.models import Spectrum
from astra.utils import log, expand_path, list_to_dict
from astra.pipelines.ferre.utils import read_ferre_headers, clip_initial_guess
from astra.pipelines.aspcap.utils import (parse_header_path, approximate_log10_microturbulence, yield_suitable_grids)
#from astra.tools.continuum import Continuum, Scalar

from typing import Iterable, Union, List, Tuple, Optional, Callable


def plan_coarse_stellar_parameter_tasks(
    spectra: Iterable[Spectrum],
    parent_dir: str,
    header_paths: Optional[Union[List[str], Tuple[str], str]] = "$MWM_ASTRA/pipelines/aspcap/synspec_dr17_marcs_header_paths.list",
    #continuum_method: Optional[Union[Continuum, str]] = Scalar,
    #continuum_kwargs: Optional[dict] = dict(method="median"),
    initial_guess_callable: Optional[Callable] = None,
    weight_path: Optional[str] = "$MWM_ASTRA/pipelines/aspcap/masks/global.mask",
    interpolation_order: Optional[int] = 3,
    f_access: Optional[int] = 0,
    n_threads: Optional[int] = 1,
    **kwargs,
):
    """
    Plan a set of FERRE executions for a coarse stellar parameter run.
    """

    if initial_guess_callable is None:
        initial_guess_callable = initial_guesses

    all_headers = {}
    for header_path in read_ferre_header_paths(header_paths):
        headers, *segment_headers = read_ferre_headers(expand_path(header_path))
        all_headers[header_path] = headers

    all_kwds = []
    for spectrum, input_initial_guess in initial_guess_callable(spectra):

        for header_path, meta, headers in yield_suitable_grids(all_headers, **input_initial_guess):

            initial_guess = clip_initial_guess(input_initial_guess, headers)

            frozen_parameters = dict()
            if meta["spectral_type"] != "BA":
                frozen_parameters.update(c_m=True, n_m=True)
                if meta["gd"] == "d" and meta["spectral_type"] == "F":
                    frozen_parameters.update(alpha_m=True)

            kwds = dict(
                spectra=spectrum,
                header_path=header_path,
                frozen_parameters=frozen_parameters,
                initial_teff=initial_guess["teff"],
                initial_logg=initial_guess["logg"], 
                initial_log10_v_sini=initial_guess["log10_v_sini"],
                initial_log10_v_micro=initial_guess["log10_v_micro"],
                initial_m_h=initial_guess["m_h"],
                initial_alpha_m=initial_guess["alpha_m"],
                initial_c_m=initial_guess["c_m"],
                initial_n_m=initial_guess["n_m"],
                initial_flags=initial_guess.get("initial_flags", 0),
                weight_path=weight_path,
                #continuum_method=continuum_method,
                #continuum_kwargs=continuum_kwargs,
            )
            kwds.update(kwargs)

            all_kwds.append(kwds)

    # Anything that has no initial guess?
    has_no_initial_guess = set([s.spectrum_id for s in spectra]).difference([ea['spectra'].spectrum_id for ea in all_kwds])

    # Bundle them together into executables based on common header paths.
    header_paths = list(set([ea["header_path"] for ea in all_kwds]))
    log.info(f"Found {len(header_paths)} unique header paths")

    grouped_task_kwds = { header_path: [] for header_path in header_paths }
    for kwds in all_kwds:
        grouped_task_kwds[kwds.pop("header_path")].append(kwds)


    return_list_of_kwds = []
    for header_path, kwds in grouped_task_kwds.items():

        grouped_task_kwds[header_path] = list_to_dict(kwds)

        short_grid_name = parse_header_path(header_path)["short_grid_name"]

        pwd = os.path.join(parent_dir, "coarse", short_grid_name)

        grouped_task_kwds[header_path].update(
            header_path=header_path,
            pwd=pwd,
            n_threads=n_threads,
            f_access=f_access,
            interpolation_order=interpolation_order,
            weight_path=weight_path,
            # Frozen parameters are common to the header path, so just set as the first value.
            frozen_parameters=grouped_task_kwds[header_path]["frozen_parameters"][0],
            **kwargs
        )
        return_list_of_kwds.append(grouped_task_kwds[header_path])

    return (return_list_of_kwds, has_no_initial_guess)

        
    # parent_dir/coarse/{grid_name}/
    # parent_dir/params/{grid_name}
    # parent_dir/abundances/{grid_name}/{element}

    '''
    # Bundle them together into executables based on common header paths.
    header_paths = list(set([ea["header_path"] for ea in task_kwds]))
    log.info(f"Found {len(header_paths)} unique header paths")

    grouped_task_kwds = { header_path: [] for header_path in header_paths }
    for kwds in task_kwds:
        header_path = kwds["header_path"]
        grouped_task_kwds[header_path].append(kwds)

    common_keys = ("header_path", "frozen_parameters")
    z = 0
    for header_path in header_paths:
        grouped_task_kwds[header_path] = list_to_dict(grouped_task_kwds[header_path])
        # same for all in this header path:
        for key in common_keys:
            grouped_task_kwds[header_path][key] = grouped_task_kwds[header_path][key][0]

        short_grid_name = header_path.split("/")[-2]

        print(header_path, z)
        while os.path.exists(expand_path(f"{parent_dir}/{short_grid_name}/{z:02d}")):
            z += 1
        
        pwd = f"{parent_dir}/{short_grid_name}/{z:02d}"
        os.makedirs(expand_path(pwd), exist_ok=True)

        # Add non-computationally relevant things.
        grouped_task_kwds[header_path].update(
            n_threads=n_threads,
            pwd=pwd,
            weight_path=weight_path,
            continuum_method=continuum_method,
            continuum_kwargs=continuum_kwargs,
            interpolation_order=interpolation_order,
            f_access=f_access
        )

    instructions = []
    for header_path, task_kwargs in grouped_task_kwds.items():
        # Resolve data products as IDs.
        task_kwargs["data_product"] = [ea.id for ea in task_kwargs["data_product"]]

        # TODO: put this functionality to a utility?
        instructions.append({
            "task_callable": "astra.contrib.aspcap.initial.aspcap_initial",
            "task_kwargs": task_kwargs,
        })

    return all_kwds, has_no_initial_guess
    '''




def read_ferre_header_paths(header_paths):
    if isinstance(header_paths, str):
        if header_paths.lower().endswith(".hdr"):
            header_paths = [header_paths]
        else:
            # Load from file.
            with open(expand_path(header_paths), "r") as fp:
                header_paths = [line.strip() for line in fp]
    return header_paths
            

            

def initial_guesses(spectrum: Spectrum) -> List[dict]:
    """
    Return a list of initial guesses for a spectrum.
    """

    defaults = dict(
        log10_v_sini=1.0,
        c_m=0,
        n_m=0,
        alpha_m=0,
        log10_v_micro=lambda logg, **_: 10**approximate_log10_microturbulence(logg)
    )

    raise NotImplementedError

