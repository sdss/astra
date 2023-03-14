import os
import json
import numpy as np
from peewee import fn
from astra.contrib.aspcap import utils
from astra.tools.continuum.base import Continuum
from astra.tools.continuum.scalar import Scalar
from typing import Union, List, Tuple, Optional, Callable, Iterable
from astra import __version__
from astra.base import task
from astra.database.astradb import DataProduct, Task
from astra.contrib.ferre.base import ferre
from astra.contrib.aspcap.models import ASPCAPInitial
from astra.utils import log, flatten, list_to_dict, deserialize, expand_path, serialize_executable
from tempfile import mkstemp
from astra.contrib.ferre.utils import read_ferre_headers
from astropy.io import fits

# FERRE v4.8.8 src trunk : /uufs/chpc.utah.edu/common/home/sdss09/software/apogee/Linux

@task
def aspcap_initial(
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
) -> Iterable[ASPCAPInitial]:
    
    yield from ferre(
        ASPCAPInitial,
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


def initial_stellar_parameters_guess_from_doppler_headers(data_product):
    """
    Return an initial guess for FERRE from Doppler given a data product.

    :param data_product:
        The data product to be analyzed with FERRE.
    """
    with fits.open(data_product.path) as image:
        for hdu in (3, 4):
            try:                            
                guess = dict(
                    teff=float(image[hdu].header["TEFF_D"]),
                    logg=float(image[hdu].header["LOGG_D"]),
                    metals=float(image[hdu].header["FEH_D"]),
                    initial_guess_source="doppler"
                )
            except:
                continue
            else:
                return (data_product, guess)


def initial_stellar_parameters_guess_from_gaia_dr3_xp(data_products):
    """
    Return an initial guess for FERRE from the Gaia DR3 XP analysis by Renae et al. (2023).

    :param data_product:
        The data product to be analyzed with FERRE.
    """
    # At the time of coding, the table is not reflected in the XP schema.
    cursor = database.execute_sql(f"""
    SELECT 
        teff_xgboost,
        logg_xgboost,
        mh_xgboost
    FROM catalogdb.xpfeh_gaia_dr3
    WHERE source_id = {data_product.source_id}
    """
    )
    r = cursor.fetchone()
    if r is not None:
        return dict(
            teff=r[0],
            logg=r[1],
            metals=r[2],
            initial_guess_source="gaia_dr3_xp"
        )
    


def initial_stellar_parameters_guess_from_apogeenet(data_products):
    """
    Return an initial guess for FERRE from APOGEENet given a data product.

    :param data_product:
        The data prodcut to be analyzed with FERRE.
    """
    from astra.contrib.apogeenet.base import ApogeeNetOutput
    source_ids = set([ea.source_id for ea in data_products])
    dps = {}
    for dp in data_products:
        dps.setdefault(dp.source_id, [])
        dps[dp.source_id].append(dp)
    
    Alias = ApogeeNetOutput.alias()
    sq = (
        Alias
        .select(
            Alias.source_id,
            fn.MAX(Alias.snr).alias("max_snr"),
        )
        .where(Alias.source_id << list(source_ids))
        .group_by(Alias.source_id)
        .alias("sq")
    )

    q = (
        ApogeeNetOutput
        .select(
            ApogeeNetOutput.source_id,
            ApogeeNetOutput.teff,
            ApogeeNetOutput.logg,
            ApogeeNetOutput.fe_h
        )
        .distinct(ApogeeNetOutput.source_id)
        .join(sq, on=(
            (ApogeeNetOutput.source_id == sq.c.source_id) &
            (ApogeeNetOutput.snr == sq.c.max_snr)
        ))
        .tuples()
    )    

    for source_id, teff, logg, metals in q:
        for data_product in dps[source_id]:
            guess = dict(
                teff=teff,
                logg=logg,
                metals=metals,
                initial_guess_source="apogeenet"
            )
            yield (data_product, guess)


def _mean_fiber(fiber, snr):
    """
    Calculate mean fiber by SNR in the way they did it for SDSS-IV.

    See https://github.com/sdss/apogee/blob/e425a6a98e3bb16cffaa86775ac507e4e887c010/pro/apogeereduce/apstar_output.pro#L229
    """
    return np.sum(fiber * snr)/np.sum(snr)



def _get_telescope_and_mean_fibers(data_product, image):
    from sdss_access import SDSSPath
    for hdu in (3, 4): 
        if image[hdu].data is not None and len(image[hdu].data) > 0:
            telescope = image[hdu].header["OBSRVTRY"].lower() + "25m"
            if data_product.filetype == "mwmStar":
                # For mean fiber we need to load it from mwmVisit. That's annoying.
                # TODO: put mean fiber into mwmStar files.
                mwmVisit_path = SDSSPath(data_product.release).full("mwmVisit", **data_product.kwargs)
                with fits.open(mwmVisit_path) as mwmVisit:
                    fiber = mwmVisit[hdu].data["FIBER"]
                    snr = mwmVisit[hdu].data["SNR"]
                    in_stack = mwmVisit[hdu].data["IN_STACK"]

                    mean_fiber = _mean_fiber(fiber[in_stack], snr[in_stack])                                    
            else:
                raise NotImplementedError                

            yield (hdu, telescope, mean_fiber)




def initial_guesses(data_products) -> List[dict]:
    """
    Return initial guesses for FERRE given a data product.
    
    :param data_product:
        The data product containing 1D spectra for a source.
    """
    # TODO: get defaults from Star (telescope, mean_fiber, etc) in a not-so-clumsy way


    defaults = dict(
        lgvsini=1.0,
        c=0,
        n=0,
        o_mg_si_s_ca_ti=0,
        log10vdop=lambda logg, **_: utils.approximate_log10_microturbulence(logg)
    )

    #callables = [
    #    initial_stellar_parameters_guess_from_apogeenet,
    #    initial_stellar_parameters_guess_from_doppler_headers,
    #]

    stellar_parameter_guesses = list(initial_stellar_parameters_guess_from_apogeenet(data_products))
    for data_product in data_products:
        stellar_parameter_guesses.append(initial_stellar_parameters_guess_from_doppler_headers(data_product))
    
    for s in stellar_parameter_guesses:
        if s is None:
            continue

        data_product, stellar_parameter_guess = s
        with fits.open(data_product.path) as image:    
            # For each HDU, check for APOGEE data, and if it's there, get the mean fiber    
            for hdu, telescope, mean_fiber in _get_telescope_and_mean_fibers(data_product, image):            
                guess = dict(
                    telescope=telescope,
                    mean_fiber=mean_fiber,
                    hdu=hdu,
                    **stellar_parameter_guess
                )
                for k, v in defaults.items():
                    try:
                        v = v(**stellar_parameter_guess)
                    except:
                        None
                    guess.setdefault(k, v)

                yield (data_product, guess)



def submit_initial_stellar_parameter_tasks(
    data_product, 
    parent_dir: str,
    header_paths: Optional[Union[List[str], Tuple[str], str]] = "$MWM_ASTRA/component_data/aspcap/synspec_dr17_marcs_header_paths.list",
    #weight_path: Optional[str] = "$MWM_ASTRA/component_data/aspcap/global_mask_v02.txt",
    weight_path: Optional[str] = "$MWM_ASTRA/component_data/aspcap/global.mask",
    continuum_method: Optional[Union[Continuum, str]] = Scalar,
    continuum_kwargs: Optional[dict] = dict(method="median"),
    initial_guess_callable: Optional[Callable] = None,
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

    instructions = create_initial_stellar_parameter_tasks(
        data_product, 
        parent_dir=parent_dir,
        header_paths=header_paths,
        weight_path=weight_path,
        continuum_method=continuum_method,
        continuum_kwargs=continuum_kwargs,
        initial_guess_callable=initial_guess_callable,
        n_threads=n_threads,
        interpolation_order=interpolation_order,
        f_access=f_access,
    )

    if not instructions:
        # TODO: this sholud probably be an operator and not a function since it skips things, uses slurm, etc.
        from airflow.exceptions import AirflowSkipException
        raise AirflowSkipException("No tasks to create")

    return utils.submit_astra_instructions(instructions, parent_dir, n_threads, slurm_kwargs)


def clip_initial_guess(initial_guess, headers, percent_epsilon=1):
    lower_limits, upper_limits = (headers["LLIMITS"], headers["ULIMITS"])
    clip = percent_epsilon * (upper_limits - lower_limits) / 100.0
    clipped_lower_limits, clipped_upper_limits = (lower_limits + clip, upper_limits - clip)

    clipped_initial_guess = {}
    sanitised_labels = [ea.lower().replace(" ", "_") for ea in headers["LABEL"]]
    for k, v in initial_guess.items():
        try:
            index = sanitised_labels.index(k)
        except:
            clipped_value = v
        else:
            clipped_value = np.clip(v, clipped_lower_limits[index], clipped_upper_limits[index])
        finally:
            if isinstance(v, (float, int)):
                clipped_value = np.round(clipped_value, 3)
            clipped_initial_guess[k] = clipped_value
    
    return clipped_initial_guess
            

def check_initial_stellar_parameter_outputs(data_product):
    """
    Check if the data products have outputs, and if there were any known FERRE timeouts.
    """

    print(type(data_product), data_product)

    if isinstance(data_product, str):
        data_product_id = json.loads(data_product)
    else:
        data_product_id = [ea.id for ea in data_product]

    N_results = (
        ASPCAPInitial
        .select()
        .where(ASPCAPInitial.data_product_id << data_product_id)
        .count()
    )
    log.info(f"Found {N_results} results from {len(set(data_product_id))} unique input data products")


    N_non_finite = (
        ASPCAPInitial
        .select()
        .where(
            (ASPCAPInitial.data_product_id << data_product_id)
        &   (ASPCAPInitial.teff < 0)
        )
        .count()
    )
    if N_non_finite > 0:
        log.warning(f"Found {N_non_finite} non-finite results")
    else:
        log.info(f"No non-finite results found")

    has_outputs_for_at_least_one_data_product = flatten(
        ASPCAPInitial
        .select(ASPCAPInitial.data_product_id)
        .distinct(ASPCAPInitial.data_product_id)
        .where(ASPCAPInitial.data_product_id << data_product_id)
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
        ASPCAPInitial
        .select()
        .where(
            (ASPCAPInitial.ferre_timeout == True)
        &   (ASPCAPInitial.data_product_id << data_product_id)
        )
        .exists()
    )
    if any_ferre_timeout:
        log.warning("Found at least one FERRE timeout")
        raise RuntimeError("ferre timeouts detected")
    else:
        log.info("No FERRE timeouts found")

    f_bad = 100 * (N_non_finite + N_missing)/len(data_product_id)
    log.warning(f"More than 1 percent of bad, non-finite, or missing results ({f_bad:.0f}%)")

    if f_bad >= 50:
        raise RuntimeError("More than 50 percent of bad, non-finite, or missing results")
    else:
        if f_bad > 0:
            log.info(f"Not throwing any warnings because the number of bad, non-finite, or missing results is less than 1% ({f_bad:.2f}%)")

    return None


def _key_for_existing_task(match_keys, kwds):
    
    items = []
    for k in match_keys:
        if k == "initial_teff":
            items.append(f"{np.round(kwds[k]):.0f}")
        else:
            if k.startswith("initial"):
                items.append(f"{np.round(kwds[k]):.2f}")
            else:
                items.append(f"{kwds[k]}")
    return "__".join(items)



def create_initial_stellar_parameter_tasks(
    data_product,
    parent_dir: str,
    header_paths: Optional[Union[List[str], Tuple[str], str]] = "$MWM_ASTRA/component_data/aspcap/synspec_dr17_marcs_header_paths.list",
    #weight_path: Optional[str] = "$MWM_ASTRA/component_data/aspcap/global_mask_v02.txt",
    weight_path: Optional[str] = "$MWM_ASTRA/component_data/aspcap/global.mask",
    continuum_method: Optional[Union[Continuum, str]] = Scalar,
    continuum_kwargs: Optional[dict] = dict(method="median"),
    initial_guess_callable: Optional[Callable] = None,
    interpolation_order: Optional[int] = 3,
    f_access: Optional[int] = 0,
    n_threads: Optional[int] = 1,
    **kwargs,
) -> List[Union[Task, int]]:
    """
    Create tasks that will use FERRE to estimate the stellar parameters given a data product.

    :param data_product:
        The input data products, or primary keys for those data products.

    :param header_paths:
        A list of FERRE header path files, or a path to a file that has one FERRE header path per line.

    :param weight_path: [optional]
        The weights path to supply to FERRE. By default this is set to the global mask used by SDSS.

    :param continuum_method: [optional]
        The method to use for continuum normalization before FERRE is executed. By default this is set to

    :param data_slice: [optional]
        Slice the input spectra and only analyze those rows that meet the slice. This is only relevant for ApStar
        input data products, where the first spectrum represents the stacked spectrum. The parmaeter is ignored
        for all other input data products.
        
    :param initial_guess_callable: [optional]
        A callable function that takes in a data product and returns a list of dictionaries of initial guesses.
        Each dictionary should contain at least the following keys:
            - telescope
            - mean_fiber
            - teff
            - logg
            - metals
            - log10vdop
            - lgvsini
            - c
            - n
            - o_mg_si_s_ca_ti

        If the callable cannot supply an initial guess for a data product, it should return None instead of a dict.
    """

    log.info(f"Data products {type(data_product)}: {data_product}")

    # Data products.
    data_product = deserialize(data_product, DataProduct)

    # Header paths.
    if isinstance(header_paths, str):
        if header_paths.lower().endswith(".hdr"):
            header_paths = [header_paths]
        else:
            # Load from file.
            with open(os.path.expandvars(os.path.expanduser(header_paths)), "r") as fp:
                header_paths = [line.strip() for line in fp]

    if continuum_method is not None:
        continuum_method = serialize_executable(continuum_method)

    grid_info = utils.parse_grid_information(header_paths)
    
    if initial_guess_callable is None:
        initial_guess_callable = initial_guesses

    # Round the initial guesses to something sensible.
    round = lambda _, d=3: float(np.round(_, d).astype(float))

    '''
    # Check for any FERRE executions, but avoid N+1 queries
    existing_outputs = prefetch(
        FerreOutput
        .select()
        .where(FerreOutput.data_product << data_product)
    )
    any_existing = (existing_outputs.count() > 0)
    if not any_existing:
        log.info(f"No existing results for these data products")
    else:
        log.info(f"Some results exist for these data products. Will check and avoid duplicate task creation.")
    '''

    try:
        existing = list(
            ASPCAPInitial
            .select()
            .where((ASPCAPInitial.data_product << data_product) & (ASPCAPInitial.teff > 0))
        )
    except:
        any_existing = False
    else:            
        join_key_str = "__"
        match_keys = ("data_product", "header_path", "frozen_parameters", "weight_path", "continuum_method", "continuum_kwargs", "hdu",
            "initial_teff", "initial_logg", "initial_metals", "initial_log10vdop")
        any_existing = len(existing) > 0

    if not any_existing:
        log.info(f"No existing results for these data products")
    else:
        log.info(f"Some results exist for these data products. Will check and avoid duplicate task creation.")
        # put in a string form for quicker matching and no N+1 queries.
        
        existing_outputs = []
        for o in existing:
            existing_outputs.append(
                _key_for_existing_task(
                    match_keys,
                    dict(zip(match_keys, [getattr(o, k) for k in match_keys]))
                )
            )

        existing_outputs = set(existing_outputs)
        
    # For each (data product, initial guess) permutation we need to create tasks based on suitable grids.
    headers = {}
    task_kwds = []
    for data_product, initial_guess in initial_guess_callable(data_product):
        for header_path, meta in utils.yield_suitable_grids(grid_info, **initial_guess):
            if header_path not in headers:
                h, *sh = read_ferre_headers(expand_path(header_path))
                headers[header_path] = h

            clipped_initial_guess = clip_initial_guess(initial_guess, headers[header_path])

            frozen_parameters = dict()
            if meta["spectral_type"] != "BA":
                frozen_parameters.update(c=True, n=True)
                if meta["gd"] == "d" and meta["spectral_type"] == "F":
                    frozen_parameters.update(o_mg_si_s_ca_ti=True)

            kwds = dict(
                data_product=data_product,
                header_path=header_path,
                frozen_parameters=frozen_parameters,
                initial_teff=round(clipped_initial_guess["teff"], 0),
                initial_logg=round(clipped_initial_guess["logg"], ),
                initial_metals=round(clipped_initial_guess["metals"]),
                initial_o_mg_si_s_ca_ti=round(clipped_initial_guess["o_mg_si_s_ca_ti"]),
                initial_lgvsini=round(clipped_initial_guess["lgvsini"]),
                initial_c=round(clipped_initial_guess["c"]),
                initial_n=round(clipped_initial_guess["n"]),
                initial_log10vdop=round(clipped_initial_guess["log10vdop"]),
                hdu=initial_guess["hdu"],
                # TODO: store where the initial guess came from?
                initial_guess_source=initial_guess["initial_guess_source"],
                weight_path=weight_path,
                continuum_method=continuum_method,
                continuum_kwargs=continuum_kwargs,
            )
            kwds.update(kwargs)

            # Keep it fast for the initial run, don't check queries.
            if any_existing:   
                '''                 
                # Check for an existing task of this kind.
                expression = (
                    (FerreOutput.data_product == data_product)
                &   (FerreOutput.header_path == header_path)
                &   (FerreOutput.frozen_parameters == frozen_parameters)
                &   (FerreOutput.weight_path == weight_path)
                &   (FerreOutput.continuum_method == continuum_method)
                &   (FerreOutput.continuum_kwargs == continuum_kwargs)
                &   (FerreOutput.hdu == kwds["hdu"])
                &   (fn.ABS(FerreOutput.initial_teff - kwds["initial_teff"]) < 1)
                &   (fn.ABS(FerreOutput.initial_logg - kwds["initial_logg"]) < 0.01)
                &   (fn.ABS(FerreOutput.initial_metals - kwds["initial_metals"]) < 0.01)
                &   (fn.ABS(FerreOutput.initial_log10vdop - kwds["initial_log10vdop"]) < 0.01)
                &   (FerreOutput.teff > 0) # check for finite output
                )
                # TODO: When we are doing individual visits, we will need to check this against spectrum meta data too.

                match = existing_outputs.where(expression)
                if match.exists():
                    log.info(f"Skipping {data_product} with these kwds because output already exists: {kwds}")
                    continue     
                '''

                if _key_for_existing_task(match_keys, kwds) in existing_outputs:
                    log.info(f"Skipping {data_product} with these kwds because output already exists: {kwds}")

                        

            task_kwds.append(kwds)
            
    # Bundle them together into executables based on common header paths.
    header_paths = list(set([ea["header_path"] for ea in task_kwds]))
    log.info(f"Found {len(header_paths)} unique header paths")

    grouped_task_kwds = { header_path: [] for header_path in header_paths }
    for kwds in task_kwds:
        header_path = kwds["header_path"]
        grouped_task_kwds[header_path].append(kwds)

    common_keys = ("header_path", "frozen_parameters")
    z = 0
    for header_path in header_paths:
        grouped_task_kwds[header_path] = list_to_dict(grouped_task_kwds[header_path])
        # same for all in this header path:
        for key in common_keys:
            grouped_task_kwds[header_path][key] = grouped_task_kwds[header_path][key][0]

        short_grid_name = header_path.split("/")[-2]

        print(header_path, z)
        while os.path.exists(expand_path(f"{parent_dir}/{short_grid_name}/{z:02d}")):
            z += 1
        
        pwd = f"{parent_dir}/{short_grid_name}/{z:02d}"
        os.makedirs(expand_path(pwd), exist_ok=True)

        # Add non-computationally relevant things.
        grouped_task_kwds[header_path].update(
            n_threads=n_threads,
            pwd=pwd,
            weight_path=weight_path,
            continuum_method=continuum_method,
            continuum_kwargs=continuum_kwargs,
            interpolation_order=interpolation_order,
            f_access=f_access
        )

    instructions = []
    for header_path, task_kwargs in grouped_task_kwds.items():
        # Resolve data products as IDs.
        task_kwargs["data_product"] = [ea.id for ea in task_kwargs["data_product"]]

        # TODO: put this functionality to a utility?
        instructions.append({
            "task_callable": "astra.contrib.aspcap.initial.aspcap_initial",
            "task_kwargs": task_kwargs,
        })
        
    return instructions


def add_penalize_chisq_values_for_initial_stellar_parameter_tasks(data_product=None):
    """
    Penalize \chi^2
    """
            
    if data_product is None:
        data_product_id = None
    elif isinstance(data_product, str):
        data_product_id = json.loads(data_product)
    else:
        data_product_id = [ea.id for ea in data_product]


    # Baseline.
    r = (
        ASPCAPInitial
        .update(penalized_log_chisq_fit=ASPCAPInitial.log_chisq_fit)
    )
    if data_product_id is not None:
        r = r.where(ASPCAPInitial.data_product_id << data_product_id)
    r = r.execute()
    log.info(f"Updated {r} rows with baseline penalized_log_chisq_fit")

    # Penalize GK-esque things at cool temperatures.
    r = (
        ASPCAPInitial
        .update(penalized_log_chisq_fit=(ASPCAPInitial.penalized_log_chisq_fit + np.log10(10)))
        .where(
            ASPCAPInitial.header_path.contains("GK_200921") 
        &   (ASPCAPInitial.teff < 3900)
        )
    )
    if data_product_id is not None:
        r = r.where(ASPCAPInitial.data_product_id << data_product_id)

    r = r.execute()
    log.info(f"Penalized {r} rows in GK grid with cool temperatures")

    # Penalize logg grid edge.
    r = (
        ASPCAPInitial
        .update(penalized_log_chisq_fit=(ASPCAPInitial.penalized_log_chisq_fit + np.log10(5)))
        .where(
            ASPCAPInitial.is_near_logg_edge
        )
    )
    if data_product_id is not None:
        r = r.where(ASPCAPInitial.data_product_id << data_product_id)
    r = r.execute()
    log.info(f"Penalized {r} rows near logg grid edge")

    # Penalize teff grid edge
    r = (
        ASPCAPInitial
        .update(penalized_log_chisq_fit=(ASPCAPInitial.penalized_log_chisq_fit + np.log10(5)))
        .where(
            ASPCAPInitial.is_near_teff_edge
        )
    )
    if data_product_id is not None:
        r = r.where(ASPCAPInitial.data_product_id << data_product_id)
    
    r = r.execute()
    log.info(f"Penalized {r} rows near teff grid edge")

    return None
    
