import os
import numpy as np
from typing import Iterable, Optional
from peewee import fn
from tqdm import tqdm
from astra import task
from astra.utils import log, list_to_dict, expand_path
from astra.models.spectrum import Spectrum
from astra.models.aspcap import FerreCoarse, FerreStellarParameters
from astra.pipelines.ferre.operator import FerreOperator, FerreMonitoringOperator
from astra.pipelines.ferre.pre_process import pre_process_ferre
from astra.pipelines.ferre.post_process import post_process_ferre
from astra.pipelines.ferre.utils import (
    parse_header_path, get_input_spectrum_primary_keys, read_control_file, read_file_with_name_and_data, read_ferre_headers,
    format_ferre_input_parameters, format_ferre_control_keywords,
)
from astra.pipelines.aspcap.utils import get_input_nml_paths, sanitise_parent_dir
from astra.pipelines.aspcap.continuum import MedianFilter
import concurrent.futures


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
            source_pk=upstream.source_pk,
            spectrum_pk=upstream.spectrum_pk,
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



def _pre_compute_continuum(coarse_result, spectrum, pre_continuum):
    try:
        # Apply continuum normalization.
        pre_computed_continuum = pre_continuum.fit(spectrum, coarse_result)
    except:
        log.exception(f"Exception when computing continuum for spectrum {spectrum} from coarse result {coarse_result}:")
        return (spectrum.spectrum_pk, None)
    else:
        return (spectrum.spectrum_pk, pre_computed_continuum)
    
def plan_stellar_parameters(
    spectra: Iterable[Spectrum],
    parent_dir: str,
    weight_path: Optional[str] = "$MWM_ASTRA/pipelines/aspcap/masks/global.mask",
    stellar_parameters_pre_continuum=MedianFilter,
    max_workers=64,
    **kwargs,
):
    """
    Plan stellar parameter executions with FERRE for some given spectra.

    Those spectra are assumed to already have `FerreCoarse` results.

    :param spectra:
        An iterable of spectra to analyze. If `None` is given then the spectra will be inferred
        from the coarse stage in `parent_dir`.
    
    :param parent_dir:
        The parent directory where these FERRE executions were planned.
    
    """

    parent_dir = sanitise_parent_dir(parent_dir)

    if spectra is None:
        # Get spectrum ids from coarse stage in parent dir.
        spectrum_pks = list(get_input_spectrum_primary_keys(f"{parent_dir}/coarse"))
        if len(spectrum_pks) == 0:
            log.warning(f"No spectrum identifiers found in {parent_dir}/coarse")
            return ([], [])
        
        # TODO: assuming all spectra are the same model type..
        model_class = Spectrum.get(spectrum_pks[0]).resolve().__class__
        spectra = (
            model_class
            .select()
            .where(model_class.spectrum_pk << spectrum_pks)
        )
    else:
        spectrum_pks = [s.spectrum_pk for s in spectra] 

    # TODO: Change FerreCoarse to store the given pwd, or the expand_path(pwd)?

    q = (
        FerreCoarse
        .select()
        .where(
            (FerreCoarse.teff.is_null(False))
        &   (FerreCoarse.logg.is_null(False))
        &   (FerreCoarse.m_h.is_null(False))
        &   (FerreCoarse.pwd.startswith(expand_path(parent_dir)))
        &   (FerreCoarse.spectrum_pk << spectrum_pks)
        )
    )
    
    lookup_spectrum_by_id = { s.spectrum_pk: s for s in spectra }

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

        coarse_results_dict.setdefault(r.spectrum_pk, [])
        coarse_results_dict[r.spectrum_pk].append(r)
    
    n_spectra_with_equally_good_results = 0
    for spectrum_pk, coarse_results in coarse_results_dict.items():
        # TODO: Should we do anything other than getting the minimum penalized rchi2?
        penalized_rchi2 = np.array([r.penalized_rchi2 for r in coarse_results])
        indices = np.argsort(penalized_rchi2)

        if len(indices) > 1 and (penalized_rchi2[indices[0]] == penalized_rchi2[indices[1]]):
            #log.warning(f"Multiple results for spectrum {spectrum_pk}: {penalized_rchi2[indices]}")
            n_spectra_with_equally_good_results += 1

        coarse_results_dict[spectrum_pk] = coarse_results[indices[0]]

    if n_spectra_with_equally_good_results:
        log.warning(f"There were {n_spectra_with_equally_good_results} spectra with multiple equally good results.")

    executor = concurrent.futures.ProcessPoolExecutor(max_workers)

    futures = []
    for coarse_result in tqdm(coarse_results_dict.values(), desc="Distributing work"):
        spectrum = lookup_spectrum_by_id[coarse_result.spectrum_pk]
        futures.append(executor.submit(_pre_compute_continuum, coarse_result, spectrum, pre_continuum))

    pre_computed_continuum = {}
    with tqdm(total=len(futures), desc="Pre-computing continuum") as pb:
        for future in concurrent.futures.as_completed(futures):
            spectrum_pk, continuum = future.result()
            pre_computed_continuum[spectrum_pk] = continuum
            pb.update()
    
    group_task_kwds = {}
    for coarse_result in tqdm(coarse_results_dict.values(), desc="Grouping results"):

        group_task_kwds.setdefault(coarse_result.header_path, [])
        spectrum = lookup_spectrum_by_id[coarse_result.spectrum_pk]

        group_task_kwds[coarse_result.header_path].append(
            dict(
                spectra=spectrum,
                pre_computed_continuum=pre_computed_continuum[spectrum.spectrum_pk],
                initial_teff=coarse_result.teff,
                initial_logg=coarse_result.logg,
                initial_m_h=coarse_result.m_h,
                initial_log10_v_sini=coarse_result.log10_v_sini,
                initial_log10_v_micro=coarse_result.log10_v_micro,
                initial_alpha_m=coarse_result.alpha_m,
                initial_c_m=coarse_result.c_m,
                initial_n_m=coarse_result.n_m,
                initial_flags=coarse_result.initial_flags,                
                upstream_pk=coarse_result.task_pk,
            )
        )

    log.info(f"Grouping spectra..")

    kwds_list = []
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

    return (kwds_list, upstream_failed)
