import os
import numpy as np
from typing import Iterable, Optional
from peewee import fn
from tqdm import tqdm
from astra import task
from astra.utils import log, list_to_dict
from astra.models.spectrum import Spectrum
from astra.models.aspcap import FerreCoarse, FerreStellarParameters
from astra.pipelines.ferre.operator import FerreOperator, FerreMonitoringOperator
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
    job_ids, executions = (
        FerreOperator(
            f"{parent_dir}/{STAGE}/", 
            **(operator_kwds or {})
        )
        .execute()
    )
    FerreMonitoringOperator(job_ids, executions).execute()
    
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
    
    ferre_kwds, upstream_failed = plan_stellar_parameters(spectra, parent_dir, weight_path, **kwargs)

    # Create the FERRE files for each execution.
    for kwd in ferre_kwds:
        pre_process_ferre(**kwd)

    # Create database entries for those with upstream failures.
    for upstream in upstream_failed:
        yield FerreStellarParameters(
            source_id=upstream.source_id,
            spectrum_id=upstream.spectrum_id,
            upstream=upstream,
            ferre_flags=upstream.ferre_flags,
            initial_flags=upstream.initial_flags,
        )



@task
def post_stellar_parameters(parent_dir, **kwargs) -> Iterable[FerreStellarParameters]:
    """
    Collect the results from FERRE and create database entries for the stellar parameter step.

    :param parent_dir:
        The parent directory where these FERRE executions were planned.
    """
    
    for pwd in map(os.path.dirname, get_input_nml_paths(parent_dir, STAGE)):
        log.info("Post-processing FERRE results in {0}".format(pwd))
        for i, kwds in enumerate(post_process_ferre(pwd)):
            yield FerreStellarParameters(**kwds)

 
def plan_stellar_parameters(
    spectra: Iterable[Spectrum],
    parent_dir: str,
    weight_path: Optional[str] = "$MWM_ASTRA/pipelines/aspcap/masks/global.mask",
    stellar_parameters_pre_continuum=MedianFilter,
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
            fn.MIN(Alias.penalized_r_chi_sq).alias("min_penalized_r_chi_sq"),
        )
        .where(
            (Alias.spectrum_id << [s.spectrum_id for s in spectra])
        &   (Alias.teff.is_null(False))
        &   (Alias.logg.is_null(False))
        &   (Alias.m_h.is_null(False))
        )
        .group_by(Alias.spectrum_id)
        .alias("sq")
    )

    q = (
        FerreCoarse
        .select()
        # Some times the same star is analysed by two grids and has the same \chi^2 (to the precision that FERRE reports)
        .join(
            sq, 
            on=(
                (FerreCoarse.spectrum_id == sq.c.spectrum_id) &
                (FerreCoarse.penalized_r_chi_sq == sq.c.min_penalized_r_chi_sq)
            )
        )
    )
    
    lookup_spectrum_by_id = { s.spectrum_id: s for s in spectra }

    log.info(f"Preparing stellar parameters")

    # We apply a pre-continuum rectification step, based on the best-fitting result from upstream.
    if stellar_parameters_pre_continuum:
        pre_continuum = stellar_parameters_pre_continuum()
    else:
        pre_continuum = None

    # Ensure only one result per spectrum ID first.
    upstream_failed, coarse_results_dict = ([], {})
    for r in q:
        # TODO: Should we do it based on other things?
        if r.flag_no_suitable_initial_guess or r.flag_spectrum_io_error:
            upstream_failed.append(r)
            continue

        coarse_results_dict.setdefault(r.spectrum_id, [])
        coarse_results_dict[r.spectrum_id].append(r)
    
    for spectrum_id, coarse_results in coarse_results_dict.items():
        if len(coarse_results) > 1:
            log.warning(f"Multiple coarse results for spectrum {spectrum_id}: {coarse_results}")

        index = np.argmin([r.r_chi_sq for r in coarse_results])
        coarse_results_dict[spectrum_id] = coarse_results[index]

    group_task_kwds, pre_computed_continuum = ({}, {})
    for coarse_result in tqdm(coarse_results_dict.values(), total=0):

        group_task_kwds.setdefault(coarse_result.header_path, [])
        spectrum = lookup_spectrum_by_id[coarse_result.spectrum_id]

        if pre_continuum is not None:
            try:
                # Apply continuum normalization.
                continuum = pre_continuum.fit(spectrum, coarse_result)
            except:
                log.exception(f"Exception for spectrum {spectrum} from coarse result {coarse_result}:")
                continue

            pre_computed_continuum[spectrum.spectrum_id] = continuum

        group_task_kwds[coarse_result.header_path].append(
            dict(
                spectra=spectrum,
                pre_computed_continuum=pre_computed_continuum.get(spectrum.spectrum_id, None),
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

    log.info(f"Grouping spectra..")

    kwds_list = []
    #spectra_with_no_coarse_result = set(spectra)
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
        #spectra_with_no_coarse_result -= set(group_task_kwds[header_path]["spectra"])

    #spectra_with_no_coarse_result = tuple(spectra_with_no_coarse_result)
    return (kwds_list, upstream_failed)
