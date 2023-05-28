import os
import numpy as np
from typing import Iterable, Optional
from peewee import fn

from astra import task
from astra.utils import log, list_to_dict
from astra.models.spectrum import Spectrum
from astra.models.aspcap import FerreCoarse, FerreStellarParameters
from astra.pipelines.ferre.operator import FerreOperator
from astra.pipelines.ferre.pre_process import pre_process_ferre
from astra.pipelines.ferre.post_process import post_process_ferre
from astra.pipelines.ferre.utils import (
    parse_header_path, read_control_file, read_file_with_name_and_data, read_ferre_headers,
    format_ferre_input_parameters, format_ferre_control_keywords
)
from astra.pipelines.aspcap.utils import get_input_nml_paths
from astra.pipelines.aspcap.continuum import MedianFilter

STAGE = "params"

@task
def stellar_parameters(
    spectra: Iterable[Spectrum],
    parent_dir: str,
    weight_path: Optional[str] = "$MWM_ASTRA/pipelines/aspcap/masks/global.mask",
    operator_kwds: Optional[dict] = None,
    **kwargs
) -> Iterable[FerreStellarParameters]:
    """
    Run the coarse stellar parameter determination step in ASPCAP.
    
    This task does the pre-processing and post-processing steps for FERRE, all in one. If you care about performance, you should
    run these steps separately and execute FERRE with a batch system.

    :param spectra:
        The spectra to be processed.
    
    :param parent_dir:
        The parent directory where these FERRE executions will be planned.    
        
    :param weight_path:
        The path to the FERRE weight file.
    """

    yield from pre_stellar_parameters(spectra, parent_dir, weight_path, **kwargs)

    # Execute ferre.
    FerreOperator(f"{parent_dir}/{STAGE}/", **(operator_kwds or {})).execute()    

    yield from post_stellar_parameters(parent_dir)


@task
def pre_stellar_parameters(
    spectra: Iterable[Spectrum],
    parent_dir: str,
    weight_path: Optional[str] = "$MWM_ASTRA/pipelines/aspcap/masks/global.mask",
    **kwargs
) -> Iterable[FerreStellarParameters]:
    """
    Prepare to run FERRE multiple times for the stellar parameter determination step.

    The `post_stellar_parameters` task will collect results from FERRE and create database entries.

    :param spectra:
        The spectra to be processed.
    
    :param parent_dir:
        The parent directory where these FERRE executions will be planned.    
    
    :param weight_path:
        The path to the FERRE weight file.
    """
    
    ferre_kwds, spectra_with_no_coarse_result = plan_stellar_parameters(spectra, parent_dir, weight_path, **kwargs)

    # Create the FERRE files for each execution.
    for kwd in ferre_kwds:
        pre_process_ferre(**kwd)

    # Create database entries for those with no initial guess?
    for spectrum in spectra_with_no_coarse_result:
        log.warning(f"Spectrum {spectrum} had no coarse result")
    #    yield FerreStellarParameters(
    #        sdss_id=spectrum.sdss_id,
    #        spectrum_id=spectrum.spectrum_id,
    #        flag_no_suitable_initial_guess=True,
    #    )
    yield from []
    

@task
def post_stellar_parameters(parent_dir, **kwargs) -> Iterable[FerreStellarParameters]:
    """
    Collect the results from FERRE and create database entries for the stellar parameter step.

    :param parent_dir:
        The parent directory where these FERRE executions were planned.
    """
    
    for pwd in map(os.path.dirname, get_input_nml_paths(parent_dir, STAGE)):
        log.info("Post-processing FERRE results in {0}".format(pwd))
        for kwds in post_process_ferre(pwd):
            yield FerreStellarParameters(**kwds)

 
def plan_stellar_parameters(
    spectra: Iterable[Spectrum],
    parent_dir: str,
    weight_path: Optional[str] = "$MWM_ASTRA/pipelines/aspcap/masks/global.mask",
    **kwargs,
):
    """
    Plan stellar parameter executions with FERRE for some given spectra.

    Those spectra are assumed to already have `FerreCoarse` results.
    """

    Alias = FerreCoarse.alias()
    sq = (
        Alias
        .select(
            Alias.spectrum_id,
            fn.MIN(Alias.ferre_log_penalized_chisq).alias("min_ferre_log_penalized_chisq"),
        )
        .where(Alias.spectrum_id << [s.spectrum_id for s in spectra])
        .group_by(Alias.spectrum_id)
        .alias("sq")
    )

    q = (
        FerreCoarse
        .select()
        .join(
            sq, 
            on=(
                (FerreCoarse.spectrum_id == sq.c.spectrum_id) &
                (FerreCoarse.ferre_log_penalized_chisq == sq.c.min_ferre_log_penalized_chisq)
            )
        )
    )
    
    lookup_spectrum_by_id = { s.spectrum_id: s for s in spectra }

    # We apply a pre-continuum rectification step, based on the best-fitting result from upstream.
    pre_continuum = MedianFilter()

    group_task_kwds = {}
    for coarse_result in q:
        group_task_kwds.setdefault(coarse_result.header_path, [])

        try:
            # Apply continuum normalization.
            spectrum = lookup_spectrum_by_id[coarse_result.spectrum_id]
            continuum = pre_continuum.fit(spectrum, coarse_result)
        except:
            log.exception(f"Exception for spectrum {spectrum} from coarse result {coarse_result}:")
            continue

        # This doesn't change the spectrum on disk, it just changes it in memory so it can be written out for FERRE.
        spectrum.flux /= continuum
        spectrum.ivar *= continuum**2

        group_task_kwds[coarse_result.header_path].append(
            dict(
                spectra=spectrum,
                initial_teff=coarse_result.teff,
                initial_logg=coarse_result.logg,
                initial_m_h=coarse_result.m_h,
                initial_log10_v_sini=coarse_result.log10_v_sini,
                initial_log10_v_micro=coarse_result.log10_v_micro,
                initial_alpha_m=coarse_result.alpha_m,
                initial_c_m=coarse_result.c_m,
                initial_n_m=coarse_result.n_m,
                initial_flags=coarse_result.initial_flags,                
                upstream_id=coarse_result.task_id,
            )
        )


    kwds_list = []
    spectra_with_no_coarse_result = set(spectra)
    for header_path in group_task_kwds.keys():

        short_grid_name = parse_header_path(header_path)["short_grid_name"]

        pwd = os.path.join(parent_dir, STAGE, short_grid_name)

        group_task_kwds[header_path] = list_to_dict(group_task_kwds[header_path])
        group_task_kwds[header_path].update(
            header_path=header_path,
            weight_path=weight_path,
            pwd=pwd,
            **kwargs
        )
        kwds_list.append(group_task_kwds[header_path])
        spectra_with_no_coarse_result -= set(group_task_kwds[header_path]["spectra"])

    spectra_with_no_coarse_result = tuple(spectra_with_no_coarse_result)
    return (kwds_list, spectra_with_no_coarse_result)