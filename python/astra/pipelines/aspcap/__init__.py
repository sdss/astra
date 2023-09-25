import os
import numpy as np
from datetime import datetime
from tempfile import mkdtemp
from typing import Optional, Iterable, List, Tuple, Callable, Union
from peewee import JOIN, fn
from tqdm import tqdm

from astra import __version__, task
from astra.utils import log, expand_path
from astra.models.aspcap import ASPCAP, FerreCoarse, FerreStellarParameters, FerreChemicalAbundances
from astra.models.spectrum import Spectrum, SpectrumMixin
from astra.pipelines.aspcap.initial import get_initial_guesses
from astra.pipelines.aspcap.coarse import coarse_stellar_parameters, post_coarse_stellar_parameters
from astra.pipelines.aspcap.stellar_parameters import stellar_parameters, post_stellar_parameters
from astra.pipelines.aspcap.abundances import abundances, get_species, post_abundances
from astra.pipelines.aspcap.utils import ABUNDANCE_RELATIVE_TO_H
        


@task
def aspcap(
    spectra: Iterable[Spectrum], 
    parent_dir: Optional[str] = None, 
    initial_guess_callable: Optional[Callable] = None,
    header_paths: Optional[Union[List[str], Tuple[str], str]] = "$MWM_ASTRA/pipelines/aspcap/synspec_dr17_marcs_header_paths.list",
    weight_path: Optional[str] = "$MWM_ASTRA/pipelines/aspcap/masks/global.mask",
    element_weight_paths: str = "$MWM_ASTRA/pipelines/aspcap/masks/elements.list",
    operator_kwds: Optional[dict] = None,
    **kwargs
) -> Iterable[ASPCAP]:
    """
    Run the ASPCAP pipeline on some spectra.
    
    .. warning:: 
        This is task for convenience. 
        
        If you want efficiency, you should use the `pre_` and `post_` tasks for each stage in the pipeline.
    
    :param spectra:
        The spectra to analyze with ASPCAP.
    
    :param parent_dir: [optional]
        The parent directory where these FERRE executions will be planned. If `None` is given then this will default
        to a temporary directory in `$MWM_ASTRA/X.Y.Z/pipelines/aspcap/`.
    
    :param initial_guess_callable: [optional]
        A callable that returns an initial guess for the stellar parameters. 
    
    :param header_paths: [optional]
        The path to a file containing the paths to the FERRE header files. This file should contain one path per line.
    
    :param weight_path: [optional]
        The path to the FERRE weight file to use during the coarse and main stellar parameter stage.

    :param element_weight_paths: [optional]
        A path containing FERRE weight files for different elements, which will be used in the chemical abundances stage.

    :param operator_kwds: [optional]
        A dictionary of keywords to supply to the `astra.pipelines.ferre.operator.FerreOperator` class.
    
    Keyword arguments
    -----------------
    All additional keyword arguments will be passed through to `astra.pipelines.ferre.pre_process.pre_process.ferre`. 
    Some handy keywords include:
    continuum_order: int = 4,
    continuum_reject: float = 0.3,
    continuum_observations_flag: int = 1,
    """

    if parent_dir is None:
        dir = expand_path(f"$MWM_ASTRA/{__version__}/pipelines/aspcap/")
        os.makedirs(dir, exist_ok=True)
        parent_dir = mkdtemp(prefix=f"{datetime.now().strftime('%Y-%m-%d')}-", dir=dir)
        os.chmod(parent_dir, 0o755)

    if initial_guess_callable is None:
        initial_guess_callable = get_initial_guesses

    # Convenience without accidentally `flatten()`ing a `ModelSelect`
    if isinstance(spectra, SpectrumMixin):
        spectra = [spectra]

    # Use the list() to make sure this is executed before other stages.
    coarse_stellar_parameter_results = list(
        coarse_stellar_parameters(
            spectra,
            parent_dir=parent_dir,
            initial_guess_callable=initial_guess_callable,
            header_paths=header_paths,
            weight_path=weight_path,
            operator_kwds=operator_kwds,
            **kwargs
        )
    )

    # Here we don't need list() because the stellar parameter results will get processed first
    # in the `create_aspcap_results` function, and then the chemical abundance results.
    # TODO: This might become a bit of a clusterfuck if the FERRE jobs fail. Maybe revisit this.
    stellar_parameter_results = list(stellar_parameters(
        spectra,
        parent_dir=parent_dir,
        weight_path=weight_path,
        operator_kwds=operator_kwds,
        **kwargs
    ))

    chemical_abundance_results = list(abundances(
        spectra,
        parent_dir=parent_dir,
        element_weight_paths=element_weight_paths,
        operator_kwds=operator_kwds,
        **kwargs
    ))
    yield from create_aspcap_results(stellar_parameter_results, chemical_abundance_results)


@task
def post_process_aspcap(parent_dir, **kwargs) -> Iterable[ASPCAP]:
    """
    Run all the post-processing steps for each ASPCAP stage.
    
    :param parent_dir:
        The parent directory where these FERRE executions will be planned. If `None` is given then this will default
        to a temporary directory in `$MWM_ASTRA/X.Y.Z/pipelines/aspcap/`.    
    """

    coarse_results = list(post_coarse_stellar_parameters(parent_dir, **kwargs))
    stellar_parameter_results = list(post_stellar_parameters(parent_dir, **kwargs))
    chemical_abundance_results = list(post_abundances(parent_dir, **kwargs))
    yield from create_aspcap_results(stellar_parameter_results, chemical_abundance_results)



@task
def create_aspcap_results(
    stellar_parameter_results: Optional[Iterable[FerreStellarParameters]] = None, 
    chemical_abundance_results: Optional[Iterable[FerreChemicalAbundances]] = None, 
    **kwargs
) -> Iterable[ASPCAP]:
    """
    Create ASPCAP results based on the results from the stellar parameter stage, and the chemical abundances stage.

    These result iterables are linked through the `FerreChemicalAbundances.upstream_pk` being equal to the
    `FerreStellarParameters.task_pk` attributes. One ASPCAP result will be created for each stellar parameter result,
    even if there are no abundances available for that stellar parameter result.

    :param stellar_parameter_results:
        An iterable of `FerreStellarParameters`.
    
    :param chemical_abundance_results:
        An iterable of `FerreChemicalAbundances`
    """

    if stellar_parameter_results is None:
            
        stellar_parameter_results = list(
            FerreStellarParameters
            .select()
            #.join(ASPCAP, JOIN.LEFT_OUTER, on=(ASPCAP.stellar_parameters_task_pk == FerreStellarParameters.task_pk))
            #.where(ASPCAP.stellar_parameters_task_pk.is_null())
        )
        '''
        if stellar_parameter_results:
            spectrum_pks = [r.spectrum_pk for r in stellar_parameter_results]
            chemical_abundance_results = list(
                FerreChemicalAbundances
                .select()
                .where(FerreChemicalAbundances.spectrum_pk.in_(spectrum_pks))
            )
        else:
            chemical_abundance_results = []
        '''
    
        chemical_abundance_results = list(
            FerreChemicalAbundances
            .select()
        )        
        
    t_coarse = (
        FerreCoarse
        .select(
            FerreCoarse.spectrum_pk,
            fn.sum(FerreCoarse.t_elapsed),        
            fn.sum(FerreCoarse.ferre_time_elapsed)
        )
        .where(FerreCoarse.spectrum_pk.in_([ea.spectrum_pk for ea in stellar_parameter_results]))
        .group_by(FerreCoarse.spectrum_pk)
        .tuples()
    )
    t_coarse = { k: v for k, *v in t_coarse }

    data, ferre_time_elapsed, t_elapsed = ({}, {}, {})

    for result in tqdm(stellar_parameter_results, desc="Collecting stellar parameters"):
        data.setdefault(result.task_pk, {})

        t_elapsed.setdefault(result.task_pk, 0)
        t_elapsed[result.task_pk] += (result.t_elapsed or 0) + (t_coarse.get(result.spectrum_pk, [0])[0] or 0)
        ferre_time_elapsed.setdefault(result.task_pk, 0)
        ferre_time_elapsed[result.task_pk] += (result.ferre_time_elapsed or 0) + (t_coarse.get(result.spectrum_pk, [0])[-1] or 0)
        
        v_sini = 10**(result.log10_v_sini or np.nan)
        e_v_sini = (result.e_log10_v_sini or np.nan) * v_sini * np.log(10)
        v_micro = 10**(result.log10_v_micro or np.nan)
        e_v_micro = (result.e_log10_v_micro or np.nan) * v_micro * np.log(10)

        data[result.task_pk].update({
            "source_pk": result.source_pk,
            "spectrum_pk": result.spectrum_pk,
            "tag": result.tag,
            "t_elapsed": t_elapsed[result.task_pk],
            "short_grid_name": result.short_grid_name,
            "teff": result.teff,
            "e_teff": result.e_teff,
            "logg": result.logg,
            "e_logg": result.e_logg,
            "v_micro": v_micro,
            "e_v_micro": e_v_micro,
            "v_sini": v_sini,
            "e_v_sini": e_v_sini,
            "m_h_atm": result.m_h,
            "e_m_h_atm": result.e_m_h,
            "alpha_m_atm": result.alpha_m,
            "e_alpha_m_atm": result.e_alpha_m,
            "c_m_atm": result.c_m,
            "e_c_m_atm": result.e_c_m,
            "n_m_atm": result.n_m,
            "e_n_m_atm": result.e_n_m,

            "initial_flags": result.initial_flags,
            "continuum_order": result.continuum_order,
            "continuum_reject": result.continuum_reject,
            "interpolation_order": result.interpolation_order,

            "snr": result.snr,
            "rchi2": result.rchi2,
            "ferre_flags": result.ferre_flags,
            "ferre_log_snr_sq": result.ferre_log_snr_sq,     
            "ferre_time_elapsed": ferre_time_elapsed[result.task_pk],
            "stellar_parameters_task_pk": result.task_pk,

            "raw_teff": result.teff,
            "raw_e_teff": result.e_teff,
            "raw_logg": result.logg,
            "raw_e_logg": result.e_logg,
            "raw_v_micro": v_micro,
            "raw_e_v_micro": e_v_micro,
            "raw_v_sini": v_sini,
            "raw_e_v_sini": e_v_sini,
            "raw_m_h_atm": result.m_h,
            "raw_e_m_h_atm": result.e_m_h,
            "raw_alpha_m_atm": result.alpha_m,
            "raw_e_alpha_m_atm": result.e_alpha_m,
            "raw_c_m_atm": result.c_m,
            "raw_e_c_m_atm": result.e_c_m,
            "raw_n_m_atm": result.n_m,
            "raw_e_n_m_atm": result.e_n_m,
        })


    # TODO: For things in chemical_abundance_results where we do not have a stellar_parameter_result..
    #       should we look to update existing ASPCAP results?

    skipped = 0
    for result in tqdm(chemical_abundance_results, desc="Collecting abundances"):
        ferre_time_elapsed.setdefault(result.upstream_pk, 0)
        ferre_time_elapsed[result.upstream_pk] += (result.ferre_time_elapsed or 0)

        if result.upstream_pk not in data:
            skipped += 1
            continue

        #data.setdefault(result.upstream_pk, {})
        species = get_species(result.weight_path)
        
        if species.lower() == "c_12_13":
            label = species.lower()
        else:
            label = f"{species.lower()}_h"

        for key in ("m_h", "alpha_m", "c_m", "n_m"):
            if not getattr(result, f"flag_{key}_frozen"):
                break
        else:
            raise ValueError(f"Can't figure out which label to use")
        
        value = getattr(result, key)
        e_value = getattr(result, f"e_{key}")

        if not ABUNDANCE_RELATIVE_TO_H[species] and value is not None:
            # [X/M] = [X/H] - [M/H]
            # [X/H] = [X/M] + [M/H]                
            value += data[result.upstream_pk]["m_h_atm"]
            e_value = np.sqrt(e_value**2 + data[result.upstream_pk]["e_m_h_atm"]**2)
            
        data[result.upstream_pk].update({
            f"{label}_task_pk": result.task_pk,
            f"{label}_rchi2": result.rchi2,
            f"{label}": value,
            f"e_{label}": e_value,
            f"raw_{label}": value,
            f"raw_e_{label}": e_value,
        })
        if hasattr(result, f"{key}_flags"):
            data[result.upstream_pk][f"{label}_flags"] = getattr(result, f"{key}_flags")
    
    if skipped:
        log.warn(
            f"Skipped {skipped} chemical abundance results because no stellar parameter result "
            f"in the accompanying argument."
        )

    for stellar_parameter_task_pk, kwds in data.items():
        yield ASPCAP(**kwds)
    