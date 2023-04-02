
import os
import json
import numpy as np
from astra.contrib.aspcap import utils
from astra.tools.continuum.base import Continuum
from typing import Union, List, Tuple, Optional, Iterable
from astra import __version__
from astra.base import task
from astra.database.astradb import DataProduct
from astra.contrib.ferre.base import ferre
from astra.contrib.aspcap.models import ASPCAPStellarParameters, ASPCAPAbundances
from astra.utils import log, flatten, list_to_dict, expand_path, serialize_executable
from peewee import fn
from astra.contrib.ferre.utils import read_ferre_headers
from astra.contrib.aspcap.utils import get_abundance_keywords



@task
def aspcap_abundances(
    header_path: str,
    pwd: str,
    data_product: Iterable[DataProduct],
    hdu: Iterable[int],
    initial_teff: Iterable[float],
    initial_logg: Iterable[float],
    initial_metals: Iterable[float],
    initial_lgvsini: Iterable[float] = None, 
    initial_log10vdop: Iterable[float] = None,
    initial_o_mg_si_s_ca_ti: Iterable[float] = None,
    initial_c: Iterable[float] = None,
    initial_n: Iterable[float] = None,
    initial_guess_source: Iterable[str] = None,
    frozen_parameters: Optional[dict] = None,
    interpolation_order: int = 3,
    weight_path: Optional[str] = None,
    lsf_shape_path: Optional[str] = None,
    lsf_shape_flag: int = 0,
    error_algorithm_flag: int = 1,
    wavelength_interpolation_flag: int = 0,
    optimization_algorithm_flag: int = 3,
    continuum_flag: int = 1,
    continuum_order: int = 4,
    continuum_segment: Optional[int] = None,
    continuum_reject: float = 0.3,
    continuum_observations_flag: int = 1,
    full_covariance: bool = True,
    pca_project: bool = False,
    pca_chi: bool = False,
    f_access: int = 0,
    f_format: int = 1,
    ferre_kwds: Optional[dict] = None,
    n_threads: int = 1,
    continuum_method: Optional[str] = None,
    continuum_kwargs: Optional[dict] = None,
    bad_pixel_flux_value: float = 1e-4,
    bad_pixel_error_value: float = 1e10,
    skyline_sigma_multiplier: float = 100,
    min_sigma_value: float = 0.05,
    spike_threshold_to_inflate_uncertainty: float = 3,
    max_spectrum_per_data_product_hdu: Optional[int] = None
) -> Iterable[ASPCAPAbundances]:
    
    yield from ferre(
        ASPCAPAbundances,
        pwd=pwd,
        header_path=header_path,
        data_product=data_product,
        hdu=hdu,
        initial_teff=initial_teff,
        initial_logg=initial_logg,
        initial_metals=initial_metals,
        initial_lgvsini=initial_lgvsini,
        initial_log10vdop=initial_log10vdop,
        initial_o_mg_si_s_ca_ti=initial_o_mg_si_s_ca_ti,
        initial_c=initial_c,
        initial_n=initial_n,
        initial_guess_source=initial_guess_source,
        frozen_parameters=frozen_parameters,
        interpolation_order=interpolation_order,
        weight_path=weight_path,
        lsf_shape_path=lsf_shape_path,
        lsf_shape_flag=lsf_shape_flag,
        error_algorithm_flag=error_algorithm_flag,
        wavelength_interpolation_flag=wavelength_interpolation_flag,
        optimization_algorithm_flag=optimization_algorithm_flag,
        continuum_flag=continuum_flag,
        continuum_order=continuum_order,
        continuum_segment=continuum_segment,
        continuum_reject=continuum_reject,
        continuum_observations_flag=continuum_observations_flag,
        full_covariance=full_covariance,
        pca_project=pca_project,
        pca_chi=pca_chi,
        f_access=f_access,
        f_format=f_format,
        ferre_kwds=ferre_kwds,
        n_threads=n_threads,
        continuum_method=continuum_method,
        continuum_kwargs=continuum_kwargs,
        bad_pixel_flux_value=bad_pixel_flux_value,
        bad_pixel_error_value=bad_pixel_error_value,
        skyline_sigma_multiplier=skyline_sigma_multiplier,
        min_sigma_value=min_sigma_value,
        spike_threshold_to_inflate_uncertainty=spike_threshold_to_inflate_uncertainty,
        max_spectrum_per_data_product_hdu=max_spectrum_per_data_product_hdu,
    )



def submit_abundance_tasks(
    data_product, 
    parent_dir: str,
    element_weight_paths: Optional[str] = "$MWM_ASTRA/component_data/aspcap/masks_ipl2/element_masks.list",
    continuum_method: Optional[Union[Continuum, str]] = "astra.contrib.aspcap.continuum.MedianFilter",
    continuum_kwargs: Optional[dict] = None,
    n_threads: Optional[int] = 1,
    interpolation_order: Optional[int] = 3,
    f_access: Optional[str] = 0,
    slurm_kwargs: Optional[dict] = None, 
    **kwargs
):
    """
    Submit tasks that will use FERRE to estimate the stellar parameters given a data product.
    """
    os.makedirs(expand_path(parent_dir), exist_ok=True)

    instructions = create_abundance_tasks(
        data_product, 
        parent_dir=parent_dir,
        element_weight_paths=element_weight_paths,
        continuum_method=continuum_method,
        continuum_kwargs=continuum_kwargs,
        n_threads=n_threads,
        interpolation_order=interpolation_order,
        f_access=f_access,
    )

    if not instructions:
        # TODO: this sholud probably be an operator and not a function since it skips things, uses slurm, etc.
        from airflow.exceptions import AirflowSkipException
        raise AirflowSkipException("No tasks to create")

    return utils.submit_astra_instructions(instructions, parent_dir, n_threads, slurm_kwargs)


def create_abundance_tasks(
    data_product,
    parent_dir,
    element_weight_paths: str = "$MWM_ASTRA/component_data/aspcap/masks_ipl2/element_masks.list",
    continuum_method: Optional[Union[Continuum, str]] = "astra.contrib.aspcap.continuum.MedianFilter",
    continuum_kwargs: Optional[dict] = None,
    interpolation_order: Optional[int] = 3,
    f_access: Optional[str] = 0,
    n_threads: Optional[int] = 1,    
):
    """
    Create FERRE tasks to estimate chemical abundances conditioned on the stellar parameters already inferred.
    """

    # Load the per element weight paths.
    with open(expand_path(element_weight_paths), "r") as fp:
        weight_paths = list(map(str.strip, fp.readlines()))
    
    # TODO: put this elsewhere
    if isinstance(data_product, str):
        data_product_ids = json.loads(data_product)
    else:
        if isinstance(data_product, (list, tuple)) and isinstance(data_product[0], int):
            data_product_ids = data_product
        else:
            data_product_ids = [ea.id for ea in data_product]    

    continuum_kwargs = continuum_kwargs or {}
    if continuum_method is not None and not isinstance(continuum_method, str):
        continuum_method = serialize_executable(continuum_method)

    fields = (
        ASPCAPStellarParameters.data_product_id,
        ASPCAPStellarParameters.hdu,
        ASPCAPStellarParameters.mjd,
        ASPCAPStellarParameters.obj,
        ASPCAPStellarParameters.plate,
        ASPCAPStellarParameters.field,
        ASPCAPStellarParameters.fiber,
        ASPCAPStellarParameters.header_path,
        ASPCAPStellarParameters.weight_path,
        ASPCAPStellarParameters.teff,
        ASPCAPStellarParameters.logg,
        ASPCAPStellarParameters.metals,
        ASPCAPStellarParameters.lgvsini,
        ASPCAPStellarParameters.log10vdop,
        ASPCAPStellarParameters.o_mg_si_s_ca_ti,
        ASPCAPStellarParameters.c,
        ASPCAPStellarParameters.n,
        # Should we use the initial_guess_source, or the ASPCAP stellar parameter point?
        ASPCAPStellarParameters.initial_guess_source,        
    )
    # TODO: Not allowing for possibility of multiple SP runs on different grids.
    distinct_fields = (
        ASPCAPStellarParameters.data_product_id,
        ASPCAPStellarParameters.hdu,
        ASPCAPStellarParameters.mjd,
        ASPCAPStellarParameters.obj,
        ASPCAPStellarParameters.plate,
        ASPCAPStellarParameters.field,
        ASPCAPStellarParameters.fiber,                
    )

    q = (
        ASPCAPStellarParameters
        .select(*fields)
        .distinct(*distinct_fields)
        .where(
            (ASPCAPStellarParameters.data_product_id << data_product_ids)
        &   (ASPCAPStellarParameters.teff > 0) 
        )
        # Use teff > 0 as some indication that things worked.
    )
    
    # Load abundance keywords on demand.
    ferre_headers = {}
    abundance_keywords = {}
    group_task_kwds = {}

    for result in q:

        # Load abundance keywords if we don't already have them.
        if result.header_path not in abundance_keywords:
            abundance_keywords[result.header_path] = {}
            headers, *segment_headers = read_ferre_headers(expand_path(result.header_path))
            ferre_headers[result.header_path] = (headers, segment_headers)
            for weight_path in weight_paths:
                species = get_species(weight_path)
                print(f"{species} from {os.path.basename(weight_path)}")                
                abundance_keywords[result.header_path][species] = get_abundance_keywords(
                    species,
                    headers["LABEL"]   
                )

        
        for weight_path in weight_paths:
            # TODO: This will get funky when we do individual visits.
            kwds = dict(
                data_product=result.data_product_id,
                hdu=result.hdu,
                initial_teff=result.teff,
                initial_logg=result.logg,
                initial_metals=result.metals,
                initial_lgvsini=result.lgvsini,
                initial_log10vdop=result.log10vdop,
                initial_o_mg_si_s_ca_ti=result.o_mg_si_s_ca_ti,
                initial_c=result.c,
                initial_n=result.n,
                initial_guess_source=result.initial_guess_source,            
            )
            key = (result.header_path, weight_path)
            group_task_kwds.setdefault(key, [])
            group_task_kwds[key].append(kwds)            

    # Create the tasks.
    z = 0
    skip_fully_frozen_keys = []
    for key in group_task_kwds.keys():
        header_path, weight_path = key
        group_task_kwds[key] = list_to_dict(group_task_kwds[key])

        species = get_species(weight_path)
        frozen_parameters, ferre_kwds = abundance_keywords[header_path][species]

        n_of_dim = ferre_headers[header_path][0]["N_OF_DIM"]
        n_frozen_dim = sum(frozen_parameters.values())
        n_free_dim = n_of_dim - n_frozen_dim
        if n_free_dim == 0:
            log.warning(f"Not creating task for {header_path} and {species} ({weight_path}) because all parameters would be frozen")
            skip_fully_frozen_keys.append(key)
            continue

        short_grid_name = f"{species}-{header_path.split('/')[-2]}"

        while os.path.exists(expand_path(f"{parent_dir}/{short_grid_name}/{z:03d}")):
            z += 1

        pwd = f"{parent_dir}/{short_grid_name}/{z:03d}"
        os.makedirs(expand_path(pwd), exist_ok=True)            

        group_task_kwds[key].update(
            header_path=header_path,
            weight_path=weight_path,
            frozen_parameters=frozen_parameters,
            ferre_kwds=ferre_kwds,
            continuum_method=continuum_method,
            continuum_kwargs=continuum_kwargs,
            pwd=pwd,
            n_threads=n_threads,
            interpolation_order=interpolation_order,
            f_access=f_access,
        )

    instructions = []
    for key, task_kwds in group_task_kwds.items():
        if key in skip_fully_frozen_keys:
            # Skip this because we 
            continue
        # TODO: put this functionality to a utility?
        instructions.append({
            "task_callable": "astra.contrib.aspcap.abundances.aspcap_abundances",
            "task_kwargs": task_kwds,
        })

    return instructions


def get_species(weight_path):
    return os.path.basename(weight_path)[:-5]


def check_abundance_outputs(data_product):
    """
    Check if the data products have outputs, and if there were any known FERRE timeouts.
    """

    # TODO: Should check per element as well..

    print(type(data_product), data_product)

    if isinstance(data_product, str):
        data_product_id = json.loads(data_product)
    else:
        data_product_id = [ea.id for ea in data_product]

    N_results = (
        ASPCAPAbundances
        .select()
        .where(ASPCAPAbundances.data_product_id << data_product_id)
        .count()
    )
    log.info(f"Found {N_results} results from {len(set(data_product_id))} unique input data products")

    N_non_finite = (
        ASPCAPAbundances
        .select()
        .where(
            (ASPCAPAbundances.data_product_id << data_product_id)
        &   (ASPCAPAbundances.teff < 0)
        )
        .count()
    )
    if N_non_finite > 0:
        log.warning(f"Found {N_non_finite} non-finite results")
    else:
        log.info(f"No non-finite results found")

    has_outputs_for_at_least_one_data_product = flatten(
        ASPCAPAbundances
        .select(ASPCAPAbundances.data_product_id)
        .distinct(ASPCAPAbundances.data_product_id)
        .where(ASPCAPAbundances.data_product_id << data_product_id)
        .tuples()
    )

    missing_data_products = set(data_product_id).difference(has_outputs_for_at_least_one_data_product)
    N_missing = len(missing_data_products)
    if N_missing > 0:
        log.warning(f"Missing results for {N_missing} data products: {missing_data_products}")
    else:
        log.info(f"All input data products have at least one result")
    
    # Number with ferre_timeout
    any_ferre_timeout = (
        ASPCAPAbundances
        .select()
        .where(
            (ASPCAPAbundances.ferre_timeout == True)
        &   (ASPCAPAbundances.data_product_id << data_product_id)
        )
        .exists()
    )
    if any_ferre_timeout:
        log.warning("Found at least one FERRE timeout")
        raise RuntimeError("ferre timeouts detected")
    else:
        log.info("No FERRE timeouts found")

    #f_bad = 100 * (N_non_finite + N_missing)/len(data_product_id)
    #if f_bad >= 1:
    #    raise RuntimeError("More than 1 percent of bad, non-finite, or missing results")
    #else:
    #    if f_bad > 0:
    #        log.info(f"Not throwing any warnings because the number of bad, non-finite, or missing results is less than 1% ({f_bad:.2f}%)")

    return None