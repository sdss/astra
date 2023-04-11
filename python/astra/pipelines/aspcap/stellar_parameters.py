import os
from typing import Iterable, Optional
from peewee import fn
from glob import glob

from astra import task
from astra.utils import expand_path, log
from astra.models import Spectrum
from astra.models.pipelines import FerreCoarse, FerreStellarParameters
from astra.pipelines.ferre import execute
from astra.pipelines.ferre.pre_process import pre_process_ferre
from astra.pipelines.ferre.post_process import post_process_ferre
from astra.pipelines.ferre.utils import parse_header_path


@task
def stellar_parameters(
    spectra: Iterable[Spectrum],
    parent_dir: str,
    weight_path: Optional[str] = "$MWM_ASTRA/pipelines/aspcap/masks/global.mask",
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

    yield from pre_process_stellar_parameters(spectra, parent_dir, weight_path, **kwargs)

    # Execute ferre.
    pwds = list(map(os.path.dirname, glob(os.path.join(expand_path(parent_dir), "*/input.nml"))))
    for pwd in pwds:
        execute(pwd)
    
    yield from post_process_stellar_parameters(parent_dir)


@task
def pre_process_stellar_parameters(
    spectra: Iterable[Spectrum],
    parent_dir: str,
    weight_path: Optional[str] = "$MWM_ASTRA/pipelines/aspcap/masks/global.mask",
    **kwargs
) -> Iterable[FerreStellarParameters]:
    """
    Prepare to run FERRE multiple times for the stellar parameter determination step.

    The `post_process_stellar_parameters` task will collect results from FERRE and create database entries.

    :param spectra:
        The spectra to be processed.
    
    :param parent_dir:
        The parent directory where these FERRE executions will be planned.    
    
    :param weight_path:
        The path to the FERRE weight file.
    """
    
    ferre_kwds = plan_stellar_parameter_executions(spectra, parent_dir, weight_path, **kwargs)

    # Create the FERRE files for each execution.
    for kwd in ferre_kwds:
        pre_process_ferre(**kwd)

    # TODO: create entries with no initial guesses?


@task
def post_process_stellar_parameters(parent_dir, **kwargs) -> Iterable[FerreStellarParameters]:
    """
    Collect the results from FERRE and create database entries for the stellar parameter step.

    :param parent_dir:
        The parent directory where these FERRE executions were planned.
    """
    
    pwds = list(map(os.path.dirname, glob(os.path.join(expand_path(parent_dir), "params/*/input.nml"))))
    for pwd in pwds:
        log.info("Post-processing FERRE results in {0}".format(pwd))
        for kwds in post_process_ferre(pwd):
            yield FerreStellarParameters(**kwds)


def plan_stellar_parameter_executions(
    spectra: Iterable[Spectrum],
    parent_dir: str,
    weight_path: Optional[str] = "$MWM_ASTRA/pipelines/aspcap/masks/global.mask",
    **kwargs,
):
    """
    Plan stellar parameter executions with FERRE for some given spectra.

    Those spectra are assumed to already have `FerreCoarse` results.
    """

    raise NotImplementedError("need to do the continuum bit")

    Alias = FerreCoarse.alias()
    sq = (
        Alias
        .select(
            Alias.spectrum_id,
            fn.MIN(Alias.ferre_log_penalized_chisq).alias("min_chisq"),
        )
        .where(Alias.spectrum_id << (s.id for s in spectra))
        .group_by(Alias.spectrum_id)
        .alias("sq")
    )

    q = (
        FerreCoarse
        .select(
            FerreCoarse.id,
            FerreCoarse.pwd,
            FerreCoarse.header_path,
            FerreCoarse.ferre_log_penalized_chisq,

            FerreCoarse.teff,
            FerreCoarse.logg,
            FerreCoarse.m_h,
            FerreCoarse.log10_v_sini,
            FerreCoarse.log10_v_micro,
            FerreCoarse.alpha_m,
            FerreCoarse.c_m,
            FerreCoarse.n_m,
            FerreCoarse.initial_flags,
        )
        .distinct(FerreCoarse.spectrum_id)
        .join(sq, on=(
            (FerreCoarse.spectrum_id == sq.c.spectrum_id) &
            (FerreCoarse.penalized_log_chisq_fit == sq.c.min_chisq)
        ))
    )

    group_task_kwds = {}
    for coarse_result in q:
        group_task_kwds.setdefault(coarse_result.header_path, [])
        group_task_kwds[coarse_result.header_path].append(
            dict(
                spectrum_id=coarse_result.spectrum_id,
                initial_teff=coarse_result.teff,
                initial_logg=coarse_result.logg,
                initial_m_h=coarse_result.m_h,
                initial_log10_v_sini=coarse_result.log10_v_sini,
                initial_log10_v_micro=coarse_result.log10_v_micro,
                initial_alpha_m=coarse_result.alpha_m,
                initial_c_m=coarse_result.c_m,
                initial_n_m=coarse_result.n_m,
                initial_flags=coarse_result.initial_flags,
                
                coarse=coarse_result,
            )
        )

    kwds_list = []
    for header_path in group_task_kwds.keys():

        short_grid_name = parse_header_path(header_path)["short_grid_name"]

        pwd = os.path.join(parent_dir, "params", short_grid_name)

        group_task_kwds[header_path].update(
            header_path=header_path,
            weight_path=weight_path,
            pwd=pwd,
            **kwargs
        )
        kwds_list.append(group_task_kwds[header_path])

    # TODO: Do anything about missing spectra?
    return kwds_list
