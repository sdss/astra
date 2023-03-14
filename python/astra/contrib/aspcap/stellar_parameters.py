
import os
import json
import numpy as np
from astra.contrib.aspcap import utils
from astra.tools.continuum.base import Continuum
from astra.tools.continuum.scalar import Scalar
from typing import Union, List, Tuple, Optional, Callable, Iterable
from astra import __version__
from astra.base import task
from astra.database.astradb import DataProduct, Task
from astra.contrib.ferre.base import ferre
from astra.contrib.aspcap.models import ASPCAPInitial, ASPCAPStellarParameters
from astra.utils import log, flatten, list_to_dict, deserialize, expand_path, serialize_executable
from tempfile import mkstemp
from astra.contrib.ferre.utils import read_ferre_headers
from astropy.io import fits
from peewee import fn
    


@task
def aspcap_stellar_parameters(
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
) -> Iterable[ASPCAPStellarParameters]:
    
    yield from ferre(
        ASPCAPStellarParameters,
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


def submit_stellar_parameter_tasks(
    data_product, 
    parent_dir: str,
    #weight_path: Optional[str] = "$MWM_ASTRA/component_data/aspcap/global_mask_v02.txt",
    weight_path: Optional[str] = "$MWM_ASTRA/component_data/aspcap/global.mask",
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

    instructions = create_stellar_parameter_tasks(
        data_product, 
        parent_dir=parent_dir,
        weight_path=weight_path,
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


def create_stellar_parameter_tasks(
    data_product,
    parent_dir,
    #weight_path: Optional[str] = "$MWM_ASTRA/component_data/aspcap/global_mask_v02.txt",
    weight_path: Optional[str] = "$MWM_ASTRA/component_data/aspcap/global.mask",
    continuum_method: Optional[Union[Continuum, str]] = "astra.contrib.aspcap.continuum.MedianFilter",
    continuum_kwargs: Optional[dict] = None,
    interpolation_order: Optional[int] = 3,
    f_access: Optional[str] = 0,
    n_threads: Optional[int] = 1,
    **kwargs
):
    """
    Create FERRE tasks to estimate stellar parameters, given the best result from the first round of
    stellar parameter determination.
    """

    log.info(f"Data product: {data_product}")

    os.makedirs(expand_path(parent_dir), exist_ok=True)

    # For each data product/HDU pair, find the best result.
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

    Alias = ASPCAPInitial.alias()
    sq = (
        Alias
        .select(
            Alias.data_product_id,
            Alias.hdu,
            fn.MIN(Alias.penalized_log_chisq_fit).alias("min_chisq"),
        )
        .where(Alias.data_product_id << data_product_ids)
        .group_by(Alias.data_product_id, Alias.hdu)
        .alias("sq")
    )

    q = (
        ASPCAPInitial
        .select(
            ASPCAPInitial.task_id,
            ASPCAPInitial.data_product_id,
            ASPCAPInitial.hdu,
            ASPCAPInitial.header_path,
            ASPCAPInitial.penalized_log_chisq_fit,
            ASPCAPInitial.teff,
            ASPCAPInitial.logg,
            ASPCAPInitial.metals,
            ASPCAPInitial.o_mg_si_s_ca_ti,
            ASPCAPInitial.lgvsini,
            ASPCAPInitial.c,
            ASPCAPInitial.n,
            ASPCAPInitial.log10vdop,
        )
        .distinct(
            ASPCAPInitial.data_product_id,
            ASPCAPInitial.hdu
        )
        .join(sq, on=(
            (ASPCAPInitial.data_product_id == sq.c.data_product_id) &
            (ASPCAPInitial.hdu == sq.c.hdu) &
            (ASPCAPInitial.penalized_log_chisq_fit == sq.c.min_chisq)
        ))
        .order_by(ASPCAPInitial.data_product_id.asc())
        .tuples()
    )

    # Group by header_path, generate tasks.
    group_task_kwds = {}
    for result in q:
        task_id, data_product_id, hdu, header_path, penalized_log_chisq_fit, teff, logg, metals, o_mg_si_s_ca_ti, lgvsini, c, n, log10vdop = result

        log.info(f'Taking {task_id} as initial guess for {data_product} (hdu={hdu}) with penalized chisq={penalized_log_chisq_fit}')

        kwds = dict(
            data_product=data_product_id,
            hdu=hdu,
            initial_teff=teff,
            initial_logg=logg,
            initial_metals=metals,
            initial_o_mg_si_s_ca_ti=o_mg_si_s_ca_ti,
            initial_lgvsini=lgvsini,
            initial_c=c,
            initial_n=n,
            initial_log10vdop=log10vdop,
            initial_guess_source=task_id,
        )
        group_task_kwds.setdefault(header_path, [])
        group_task_kwds[header_path].append(kwds)
    

    z = 0
    for header_path in group_task_kwds.keys():
        group_task_kwds[header_path] = list_to_dict(group_task_kwds[header_path])

        short_grid_name = header_path.split("/")[-2]

        while os.path.exists(expand_path(f"{parent_dir}/{short_grid_name}/{z:03d}")):
            z += 1
        
        pwd = f"{parent_dir}/{short_grid_name}/{z:03d}"
        os.makedirs(expand_path(pwd), exist_ok=True)

        group_task_kwds[header_path].update(   
            header_path=header_path,     
            weight_path=weight_path,
            continuum_method=continuum_method,
            continuum_kwargs=continuum_kwargs,
            pwd=pwd,
            n_threads=n_threads,
            interpolation_order=interpolation_order,
            f_access=f_access,
        )

    instructions = []
    for header_path, task_kwds in group_task_kwds.items():
        # TODO: put this functionality to a utility?
        instructions.append({
            "task_callable": "astra.contrib.aspcap.stellar_parameters.aspcap_stellar_parameters",
            "task_kwargs": task_kwds,
        })

    return instructions


def check_stellar_parameter_outputs(data_product):
    """
    Check if the data products have outputs, and if there were any known FERRE timeouts.
    """

    print(type(data_product), data_product)

    if isinstance(data_product, str):
        data_product_id = json.loads(data_product)
    else:
        data_product_id = [ea.id for ea in data_product]

    N_results = (
        ASPCAPStellarParameters
        .select()
        .where(ASPCAPStellarParameters.data_product_id << data_product_id)
        .count()
    )
    log.info(f"Found {N_results} results from {len(set(data_product_id))} unique input data products")


    N_non_finite = (
        ASPCAPStellarParameters
        .select()
        .where(
            (ASPCAPStellarParameters.data_product_id << data_product_id)
        &   (ASPCAPStellarParameters.teff < 0)
        )
        .count()
    )
    if N_non_finite > 0:
        log.warning(f"Found {N_non_finite} non-finite results")
    else:
        log.info(f"No non-finite results found")

    has_outputs_for_at_least_one_data_product = flatten(
        ASPCAPStellarParameters
        .select(ASPCAPStellarParameters.data_product_id)
        .distinct(ASPCAPStellarParameters.data_product_id)
        .where(ASPCAPStellarParameters.data_product_id << data_product_id)
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
        ASPCAPStellarParameters
        .select()
        .where(
            (ASPCAPStellarParameters.ferre_timeout == True)
        &   (ASPCAPStellarParameters.data_product_id << data_product_id)
        )
        .exists()
    )
    if any_ferre_timeout:
        log.warning("Found at least one FERRE timeout")
        raise RuntimeError("ferre timeouts detected")
    else:
        log.info("No FERRE timeouts found")

    f_bad = 100 * (N_non_finite + N_missing)/len(data_product_id)
    if f_bad >= 1:
        raise RuntimeError("More than 1 percent of bad, non-finite, or missing results")
    else:
        if f_bad > 0:
            log.info(f"Not throwing any warnings because the number of bad, non-finite, or missing results is less than 1% ({f_bad:.2f}%)")

    return None
