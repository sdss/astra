import os
import numpy as np
from typing import Iterable, Optional
from peewee import fn
from glob import glob

from astra import task
from astra.utils import expand_path, log, list_to_dict
from astra.models.spectrum import Spectrum
from astra.models.aspcap import FerreStellarParameters, FerreChemicalAbundances
from astra.pipelines.ferre.operator import FerreOperator, FerreMonitoringOperator
from astra.pipelines.ferre.pre_process import pre_process_ferre
from astra.pipelines.ferre.post_process import post_process_ferre
from astra.pipelines.ferre.utils import (read_ferre_headers, parse_header_path, get_input_spectrum_primary_keys)
from astra.pipelines.aspcap.utils import (get_input_nml_paths, get_abundance_keywords)

STAGE = "abundances"

@task
def abundances(
    spectra: Iterable[Spectrum],
    parent_dir: str,
    element_weight_paths: str,
    operator_kwds: Optional[dict] = None,
    **kwargs
) -> Iterable[FerreChemicalAbundances]:
    """
    Run the abundance stage in ASPCAP.
    
    This task does the pre-processing and post-processing steps for FERRE, all in one. If you care about performance, you should
    run these steps separately and execute FERRE with a batch system.

    :param spectra:
        The spectra to be processed.
    
    :param parent_dir:
        The parent directory where these FERRE executions will be planned.    
    """

    yield from pre_abundances(
        spectra, 
        parent_dir, 
        element_weight_paths,
        **kwargs
    )

    job_ids, executions = (
        FerreOperator(
            f"{parent_dir}/{STAGE}/", 
            **(operator_kwds or {})
        )
        .execute()
    )
    FerreMonitoringOperator(job_ids, executions).execute()
    
    yield from post_abundances(parent_dir)


@task
def pre_abundances(
    spectra: Iterable[Spectrum], 
    parent_dir: str, 
    element_weight_paths: str = "$MWM_ASTRA/pipelines/aspcap/masks/elements.list",
    **kwargs
) -> Iterable[FerreChemicalAbundances]:
    """
    Prepare to run FERRE multiple times for the abundance determination stage.

    The `post_abundances` task will collect results from FERRE and create database entries.

    :param spectra:
        The spectra to be processed.
    
    :param parent_dir:
        The parent directory where these FERRE executions will be planned.    
    """
    
    ferre_kwds, spectra_with_no_stellar_parameters = plan_abundances(
        spectra,
        parent_dir,
        element_weight_paths,
        **kwargs
    )
    # In other stages we would yield back a database result with a flag indicating that
    # there was nothing that could be done, but here we won't. That's because the final
    # ASPCAP table is built by using the `FerreStellarParameters` as a reference table,
    # not the `FerreChemicalAbundances` table. The chemical abundances are a bonus.
    if spectra_with_no_stellar_parameters:
        log.warning(
            f"There were {len(spectra_with_no_stellar_parameters)} spectra with no suitable stellar parameters."
        )

    # Create the FERRE files for each execution.
    group = {}
    for kwd in ferre_kwds:
        pre_process_ferre(**kwd)

        pwd = kwd["pwd"].rstrip("/")
        group_dir = "/".join(pwd.split("/")[:-1])
        group.setdefault(group_dir, [])
        group[group_dir].append(pwd[1 + len(group_dir):] + "/input.nml")
        
    # Create a parent input_list.nml file to use with the ferre.x -l flag.
    for pwd, items in group.items():
        input_list_path = f"{pwd}/input_list.nml"
        log.info(f"Created grouped FERRE input file with {len(items)} dirs: {input_list_path}")
        with open(expand_path(input_list_path), "w") as fp:
            # Sometimes `wc` would not give the right amount of lines in a file, so we add a \n to the end
            # https://unix.stackexchange.com/questions/314256/wc-l-not-returning-correct-value
            fp.write("\n".join(items) + "\n")
    
    yield from []


@task
def post_abundances(parent_dir, skip_pixel_arrays=True, **kwargs) -> Iterable[FerreChemicalAbundances]:
    """
    Collect the results from FERRE and create database entries for the abundance step.

    :param parent_dir:
        The parent directory where these FERRE executions were planned.
    """    

    # Note the "/*" after STAGE because of the way folders are structured for abundances
    # And we use the `ref_dir` because it was executed from the parent folder.
    for dir in map(os.path.dirname, get_input_nml_paths(parent_dir, f"{STAGE}/*")):
        log.info(f"Post-processing FERRE results in {dir}")
        ref_dir = os.path.dirname(dir)
        for kwds in post_process_ferre(dir, ref_dir, skip_pixel_arrays=skip_pixel_arrays, **kwargs):
            yield FerreChemicalAbundances(**kwds)    


def plan_abundances(
    spectra: Iterable[Spectrum],
    parent_dir: str,
    element_weight_paths: str,
    continuum_order: Optional[int] = -1,
    continuum_flag: Optional[int] = 0,
    continuum_observations_flag: Optional[int] = 0,
    **kwargs,
):
    """
    Plan abundance executions with FERRE for some given spectra, which are assumed to already have `FerreStellarParameter` results.

    In the abundances stage we keep the continuum fixed to what was found from the stellar parameter stage. That's why the
    defaults are set for `continuum_order`, `continuum_flag`, and `continuum_observations_flag`.

    :param spectra:
        The spectra to be processed.
    
    :param parent_dir:
        The parent directory where these FERRE executions will be planned.
    
    :param element_weight_paths:
        A path containing the masks to supply per element.
    """

    with open(expand_path(element_weight_paths), "r") as fp:
        weight_paths = list(map(str.strip, fp.readlines()))


    if spectra is None:
        # Get spectrum ids from params stage in parent dir.
        spectrum_pks = list(get_input_spectrum_primary_keys(f"{parent_dir}/params"))
        if len(spectrum_pks) == 0:
            log.warning(f"No spectrum identifiers found in {parent_dir}/params")
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


    Alias = FerreStellarParameters.alias()
    sq = (
        Alias
        .select(
            Alias.spectrum_pk.alias("spectrum_pk"),
            fn.MIN(Alias.penalized_rchi2).alias("min_penalized_rchi2"),
        )
        .where(Alias.spectrum_pk << spectrum_pks)
        .group_by(Alias.spectrum_pk)
        .alias("sq")
    )

    q = (
        FerreStellarParameters
        .select()
        # Only get one result per spectrum.
        .where(
            FerreStellarParameters.penalized_rchi2.is_null(False)
        &   (~FerreStellarParameters.flag_ferre_fail)
        &   (~FerreStellarParameters.flag_no_suitable_initial_guess)
        &   (~FerreStellarParameters.flag_missing_model_flux)
            # Don't calculate abundances for stellar parameters that are on the grid edge of TEFF/LOGG/METALS
        &   (~FerreStellarParameters.flag_teff_grid_edge_bad)
        &   (~FerreStellarParameters.flag_logg_grid_edge_bad)
        &   (~FerreStellarParameters.flag_m_h_grid_edge_bad)
        )
        .join(
            sq, 
            on=(
                (FerreStellarParameters.spectrum_pk == sq.c.spectrum_pk) &
                (FerreStellarParameters.penalized_rchi2 == sq.c.min_penalized_rchi2)
            )
        )
        # We will only get one result per spectrum, but we'll do it by recency.
        .order_by(FerreStellarParameters.task_pk.desc())
    )

    # Load abundance keywords on demand.
    ferre_headers, abundance_keywords = ({}, {})
    lookup_spectrum_by_primary_key = { s.spectrum_pk: s for s in spectra }

    done, group_task_kwds, pre_computed_continuum = ([], {}, {})
    for result in q:
        if result.spectrum_pk in done:
            continue
        done.append(result.spectrum_pk)
        group_task_kwds.setdefault(result.header_path, [])

        if result.header_path not in abundance_keywords:
            abundance_keywords[result.header_path] = {}
            headers, *segment_headers = ferre_headers[result.header_path] = read_ferre_headers(result.header_path)
            for weight_path in weight_paths:
                species = get_species(weight_path)
                frozen_parameters, ferre_kwds = get_abundance_keywords(species, headers["LABEL"])
                abundance_keywords[result.header_path][species] = (weight_path, frozen_parameters, ferre_kwds)
                
        spectrum = lookup_spectrum_by_primary_key[result.spectrum_pk]

        # Apply continuum normalization, where we are just going to fix the observed
        # spectrum to the best-fitting model spectrum from the upstream task.
        try:
            pre_computed_continuum[result.spectrum_pk]
        except KeyError:
            pre_computed_continuum[result.spectrum_pk] = result.unmask(
                (result.rectified_model_flux/result.model_flux)
            /   (result.rectified_flux/result.ferre_flux)
            )
        
        group_task_kwds[result.header_path].append(
            dict(
                spectra=spectrum,
                pre_computed_continuum=pre_computed_continuum[result.spectrum_pk],
                initial_teff=result.teff,
                initial_logg=result.logg,
                initial_m_h=result.m_h,
                initial_log10_v_sini=result.log10_v_sini,
                initial_log10_v_micro=result.log10_v_micro,
                initial_alpha_m=result.alpha_m,
                initial_c_m=result.c_m,
                initial_n_m=result.n_m,
                upstream_pk=result.task_pk,
            )
        )

    """
    The file structure for the abundances stage is:
    
    PARENT/abundances/
    PARENT/abundances/Mg_d/
    PARENT/abundances/Mg_d/input.nml
    PARENT/abundances/Mg_d/flux.input
    PARENT/abundances/Mg_d/e_flux.input
    PARENT/abundances/Mg_d/Al/input.nml
    PARENT/abundances/Mg_d/Al/parameters.input
    PARENT/abundances/Mg_d/Al/parameters.output

    # These two should be the same if we are not doing normalisation
    PARENT/abundances/Mg_d/Al/model_flux.output
    PARENT/abundances/Mg_d/Al/rectified_model_flux.output
    
    """
    extra_kwds = dict(
        continuum_order=continuum_order,
        continuum_flag=continuum_flag,
        continuum_observations_flag=continuum_observations_flag,
    )
    extra_kwds.update(kwargs)

    kwds_list = []
    spectra_with_no_stellar_parameters = set(spectra)
    for header_path in group_task_kwds.keys():

        grid_kwds = list_to_dict(group_task_kwds[header_path])
        short_grid_name = parse_header_path(header_path)["short_grid_name"]

        for i, (species, details) in enumerate(abundance_keywords[header_path].items()):
            weight_path, frozen_parameters, ferre_kwds = details
            pwd = os.path.join(parent_dir, STAGE, short_grid_name, species)
            kwds = grid_kwds.copy()
            kwds.update(
                pwd=pwd,
                header_path=header_path,
                weight_path=weight_path,
                frozen_parameters=frozen_parameters,
                ferre_kwds=ferre_kwds,
                # In the chemical abundances stage we avoid repeating the flux/e_flux files everywhere
                reference_pixel_arrays_for_abundance_run=True,
                # Only write the flux arrays to the parent folder on the first run of this header path
                write_input_pixel_arrays=(i == 0)
            )
            kwds.update(extra_kwds)
            
            if all(frozen_parameters.get(ln, False) for ln in ferre_headers[kwds["header_path"]][0]["LABEL"]):
                log.warning(f"Ignoring {species} species on grid {short_grid_name} because all parameters are frozen")
                continue
            kwds_list.append(kwds)
        
        spectra_with_no_stellar_parameters -= set(grid_kwds["spectra"])
        
    spectra_with_no_stellar_parameters = tuple(spectra_with_no_stellar_parameters)

    return (kwds_list, spectra_with_no_stellar_parameters)


def get_species(weight_path):
    return os.path.basename(weight_path)[:-5]
