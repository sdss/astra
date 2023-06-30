import os
import numpy as np
from typing import Iterable, Optional
from peewee import fn
from glob import glob

from astra import task
from astra.utils import expand_path, log, list_to_dict
from astra.models.spectrum import Spectrum
from astra.models.aspcap import FerreStellarParameters, FerreChemicalAbundances
from astra.pipelines.ferre.operator import FerreOperator
from astra.pipelines.ferre.pre_process import pre_process_ferre
from astra.pipelines.ferre.post_process import post_process_ferre
from astra.pipelines.ferre.utils import (read_ferre_headers, parse_header_path)
from astra.pipelines.aspcap.utils import (get_input_nml_paths, get_abundance_keywords)

STAGE = "abundances"

@task
def abundances(
    spectra: Iterable[Spectrum],
    parent_dir: str,
    element_weight_paths: str = "$MWM_ASTRA/pipelines/aspcap/masks/elements.list",
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

    # Execute ferre.
    FerreOperator(
        f"{parent_dir}/{STAGE}/", 
        **(operator_kwds or {})
    ).execute()    


    raise NotImplementedError("need the ferre.x -l switch")
    # Execute ferre.
    print("Switch to using the FerreOperator here, which will use the ferre.x -l thing")
    for path in get_input_nml_paths(parent_dir):
        execute(path)
    
    yield from post_abundances(parent_dir)


@task
def pre_abundances(
    spectra: Iterable[Spectrum], 
    parent_dir: str, 
    element_weight_paths: str,
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
    assert not spectra_with_no_stellar_parameters

    # Create the FERRE files for each execution.
    group = {}
    for kwd in ferre_kwds:
        pre_process_ferre(**kwd)

        pwd = kwd["pwd"].rstrip("/")
        group_dir = "/".join(pwd.split("/")[:-1])
        group.setdefault(group_dir, [])
        group[group_dir].append(pwd[1 + len(group_dir):] + "/input.nml")

        
    # Create a parent input_list.nml file to use with the -l flag.
    for pwd, items in group.items():
        input_list_path = f"{pwd}/input_list.nml"
        log.info(f"Created grouped FERRE input file with {len(items)} dirs: {input_list_path}")
        with open(input_list_path, "w") as fp:
            fp.write("\n".join(items))
    
    
    
    yield from []


@task
def post_abundances(parent_dir, **kwargs) -> Iterable[FerreChemicalAbundances]:
    """
    Collect the results from FERRE and create database entries for the abundance step.

    :param parent_dir:
        The parent directory where these FERRE executions were planned.
    """
    
    for pwd in map(os.path.dirname, get_input_nml_paths(parent_dir, STAGE)):
        log.info("Post-processing FERRE results in {0}".format(pwd))
        for kwds in post_process_ferre(pwd):
            yield FerreChemicalAbundances(**kwds)    


def plan_abundances(
    spectra: Iterable[Spectrum],
    parent_dir: str,
    element_weight_paths: str,
    **kwargs,
):
    """
    Plan abundance executions with FERRE for some given spectra.

    Those spectra are assumed to already have `FerreStellarParameter` results.

    :param spectra:
        The spectra to be processed.
    
    :param parent_dir:
        The parent directory where these FERRE executions will be planned.
    
    :param element_weight_paths:
        A path containing the masks to supply per element.
    """

    with open(expand_path(element_weight_paths), "r") as fp:
        weight_paths = list(map(str.strip, fp.readlines()))

    Alias = FerreStellarParameters.alias()
    sq = (
        Alias
        .select(
            Alias.spectrum_id,
            fn.MIN(Alias.ferre_log_chisq).alias("min_ferre_log_chisq"),
        )
        .where(Alias.spectrum_id << [s.spectrum_id for s in spectra])
        .group_by(Alias.spectrum_id)
        .alias("sq")
    )

    q = (
        FerreStellarParameters
        .select()
        .where(
            FerreStellarParameters.ferre_log_chisq.is_null(False)
        &   (~FerreStellarParameters.flag_ferre_fail)
        &   (~FerreStellarParameters.flag_no_suitable_initial_guess)
        &   (~FerreStellarParameters.flag_missing_model_flux)
        )
        .join(
            sq, 
            on=(
                (FerreStellarParameters.spectrum_id == sq.c.spectrum_id) &
                (FerreStellarParameters.ferre_log_chisq == sq.c.min_ferre_log_chisq)
            )
        )
    )
    
    # Load abundance keywords on demand.
    ferre_headers = {}
    abundance_keywords = {}
    lookup_spectrum_by_id = { s.spectrum_id: s for s in spectra }

    group_task_kwds = {}
    for result in q:
        group_task_kwds.setdefault(result.header_path, [])

        if result.header_path not in abundance_keywords:
            abundance_keywords[result.header_path] = {}
            headers, *segment_headers = ferre_headers[result.header_path] = read_ferre_headers(result.header_path)
            for weight_path in weight_paths:
                species = get_species(weight_path)
                print(f"{species} from {os.path.basename(weight_path)}")                
                frozen_parameters, ferre_kwds = get_abundance_keywords(species, headers["LABEL"])
                abundance_keywords[result.header_path][species] = (weight_path, frozen_parameters, ferre_kwds)
                
        spectrum = lookup_spectrum_by_id[result.spectrum_id]

        # Apply continuum normalization, where we are just going to fix the observed
        # spectrum to the best-fitting model spectrum from the upstream task.
        continuum = result.unmask(
            (result.rectified_model_flux/result.model_flux)
        /   (result.rectified_flux/result.ferre_flux)
        )
        
        # This doesn't change the spectrum on disk, it just changes it in memory so it can be written out for FERRE.
        spectrum.flux /= continuum
        spectrum.ivar *= continuum**2

        group_task_kwds[result.header_path].append(
            dict(
                spectra=spectrum,
                initial_teff=result.teff,
                initial_logg=result.logg,
                initial_m_h=result.m_h,
                initial_log10_v_sini=result.log10_v_sini,
                initial_log10_v_micro=result.log10_v_micro,
                initial_alpha_m=result.alpha_m,
                initial_c_m=result.c_m,
                initial_n_m=result.n_m,
                upstream_id=result.task_id,
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
            # Ignore additional keyword arguments?
            log.warning(f"Ignoring additional keyword arguments at abundances stage.")
            kwds_list.append(kwds)
        
        spectra_with_no_stellar_parameters -= set(grid_kwds["spectra"])
        
    spectra_with_no_stellar_parameters = tuple(spectra_with_no_stellar_parameters)

    return (kwds_list, spectra_with_no_stellar_parameters)


def get_species(weight_path):
    return os.path.basename(weight_path)[:-5]
