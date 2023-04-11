import os
import numpy as np
from glob import glob
from astra import task
from astra.models import Spectrum
from astra.models.pipelines import FerreCoarse
from astra.utils import log, expand_path, list_to_dict
from astra.pipelines.ferre import execute
from astra.pipelines.ferre.pre_process import pre_process_ferre
from astra.pipelines.ferre.post_process import post_process_ferre
from astra.pipelines.ferre.utils import (parse_header_path, read_ferre_headers, clip_initial_guess)
from astra.pipelines.aspcap.utils import (approximate_log10_microturbulence, yield_suitable_grids)
#from astra.tools.continuum import Continuum, Scalar

from typing import Iterable, Union, List, Tuple, Optional, Callable


@task
def coarse_stellar_parameters(
    spectra: Iterable[Spectrum],
    parent_dir: str,
    initial_guess_callable: Optional[Callable] = None,
    header_paths: Optional[Union[List[str], Tuple[str], str]] = "$MWM_ASTRA/pipelines/aspcap/synspec_dr17_marcs_header_paths.list",
    weight_path: Optional[str] = "$MWM_ASTRA/pipelines/aspcap/masks/global.mask",
    **kwargs
) -> Iterable[FerreCoarse]:
    """
    Run the coarse stellar parameter determination step in ASPCAP.
    
    This task does the pre-processing and post-processing steps for FERRE, all in one. If you care about performance, you should
    run these steps separately and execute FERRE with a batch system.

    :param spectra:
        The spectra to be processed.
    
    :param parent_dir:
        The parent directory where these FERRE executions will be planned.    
    
    :param initial_guess_callable:
        A callable that returns an initial guess for the stellar parameters. 
    
    :param header_paths:
        The path to a file containing the paths to the FERRE header files. This file should contain one path per line.
    
    :param weight_path:
        The path to the FERRE weight file.
    """

    yield from pre_process_coarse_stellar_parameters(
        spectra,
        parent_dir,
        initial_guess_callable,
        header_paths,
        weight_path,
        **kwargs
    )

    # Execute ferre.
    pwds = list(map(os.path.dirname, glob(os.path.join(expand_path(parent_dir), "*/input.nml"))))
    for pwd in pwds:
        execute(pwd)
    
    yield from post_process_coarse_stellar_parameters(parent_dir, **kwargs)


@task
def pre_process_coarse_stellar_parameters(
    spectra: Iterable[Spectrum],
    parent_dir: str,
    initial_guess_callable: Optional[Callable] = None,
    header_paths: Optional[Union[List[str], Tuple[str], str]] = "$MWM_ASTRA/pipelines/aspcap/synspec_dr17_marcs_header_paths.list",
    weight_path: Optional[str] = "$MWM_ASTRA/pipelines/aspcap/masks/global.mask",
    **kwargs
) -> Iterable[FerreCoarse]:
    """
    Prepare to run FERRE multiple times for the coarse stellar parameter determination step.

    This task will only create `FerreCoarse` database entries for spectra that had no suitable initial guess.
    The `post_process_coarse_stellar_parameters` task will collect results from FERRE and create database entries.

    :param spectra:
        The spectra to be processed.
    
    :param parent_dir:
        The parent directory where these FERRE executions will be planned.    
    
    :param initial_guess_callable:
        A callable that returns an initial guess for the stellar parameters. 
    
    :param header_paths:
        The path to a file containing the paths to the FERRE header files. This file should contain one path per line.
    
    :param weight_path:
        The path to the FERRE weight file.
    """
    
    ferre_kwds, spectra_with_no_initial_guess = plan_coarse_stellar_parameter_executions(
        spectra,
        parent_dir,
        header_paths,
        initial_guess_callable,
        weight_path,
        **kwargs,
    )

    # Create the FERRE files for each execution.
    for kwd in ferre_kwds:
        pre_process_ferre(**kwd)

    # Create database entries for those with no initial guess.
    for spectrum in spectra_with_no_initial_guess:
        yield FerreCoarse(
            spectrum_id=spectrum.id,
            source_id=spectrum.source_id,
            flag_no_suitable_initial_guess=True,
        )


@task
def post_process_coarse_stellar_parameters(parent_dir, **kwargs) -> Iterable[FerreCoarse]:
    """
    Collect the results from FERRE and create database entries for the coarse stellar parameter determination step.

    :param parent_dir:
        The parent directory where these FERRE executions were planned.
    """

    pwds = list(map(os.path.dirname, glob(os.path.join(expand_path(parent_dir), "coarse/*/input.nml"))))
    for pwd in pwds:
        log.info("Post-processing FERRE results in {0}".format(pwd))
        for kwds in post_process_ferre(pwd):
            result = FerreCoarse(**kwds)
            penalize_coarse_stellar_parameter_result(result)
            yield result


def penalize_coarse_stellar_parameter_result(result: FerreCoarse):
    """
    Penalize the coarse stellar parameter result if it is not a good fit.
    """

    # Penalize GK-esque things at cool temperatures.
    result.ferre_log_penalized_chisq = 0 + result.ferre_log_chisq
    if result.teff < 3900 and "GK_200921" in result.header_path:
        result.ferre_log_penalized_chisq += np.log10(10)
    
    # TODO: check if this should be for warn or bad.
    if result.flag_logg_grid_edge_warn:
        result.ferre_log_penalized_chisq += np.log10(5)

    if result.flag_teff_grid_edge_warn:
        result.ferre_log_penalized_chisq += np.log10(5)
    
    return None


        
def plan_coarse_stellar_parameter_executions(
    spectra: Iterable[Spectrum],
    parent_dir: str,
    header_paths: Optional[Union[List[str], Tuple[str], str]] = "$MWM_ASTRA/pipelines/aspcap/synspec_dr17_marcs_header_paths.list",
    #continuum_method: Optional[Union[Continuum, str]] = Scalar,
    #continuum_kwargs: Optional[dict] = dict(method="median"),
    initial_guess_callable: Optional[Callable] = None,
    weight_path: Optional[str] = "$MWM_ASTRA/pipelines/aspcap/masks/global.mask",
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
    all_spectrum_ids_with_at_least_one_initial_guess = []

    for spectrum, input_initial_guess in initial_guess_callable(spectra):

        for header_path, meta, headers in yield_suitable_grids(all_headers, **input_initial_guess):
            all_spectrum_ids_with_at_least_one_initial_guess.append(spectrum.spectrum_id)

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

            all_kwds.append(kwds)

    # Anything that has no initial guess?
    spectra_with_no_initial_guess = [
        s for s in spectra \
            if s.spectrum_id not in all_spectrum_ids_with_at_least_one_initial_guess
    ]

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
            weight_path=weight_path,
            # Frozen parameters are common to the header path, so just set as the first value.
            frozen_parameters=grouped_task_kwds[header_path]["frozen_parameters"][0],
            **kwargs
        )
        return_list_of_kwds.append(grouped_task_kwds[header_path])

    return (return_list_of_kwds, spectra_with_no_initial_guess)




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

