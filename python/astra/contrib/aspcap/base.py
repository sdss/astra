
import os
import json
import numpy as np
from astra.contrib.aspcap import continuum, utils
from astra.tools.continuum.base import Continuum
from astra.tools.continuum.scalar import Scalar
from typing import Union, List, Tuple, Optional, Callable
from collections import OrderedDict

from astra import log, __version__
from astra.base import TaskInstance, Parameter, TupleParameter
from astra.utils import bundler, deserialize, expand_path, serialize_executable
from astra.database.astradb import (
    database,
    FerreOutput,
    AspcapOutput,
    Output,
    TaskOutput,
    Source,
    Task,
    Bundle,
    TaskBundle,
    DataProduct,
    TaskInputDataProducts,
)
from peewee import fn
from astra.contrib.ferre.base import Ferre
from astra.contrib.ferre.utils import read_ferre_headers, sanitise
from astra.contrib.ferre.bitmask import ParamBitMask
from astra.sdss.datamodels.pipeline import create_pipeline_product

from astropy.io import fits
from astra.utils import list_to_dict

FERRE_TASK_NAME = serialize_executable(Ferre)
FERRE_DEFAULTS = Ferre.get_defaults()


def initial_guess_doppler(
    data_product: DataProduct,
    source: Optional[Source] = None,
    star = None,
) -> dict:
    """
    Return an initial guess for FERRE from Doppler given a data product.

    :param data_product:
        The data product to be analyzed with FERRE.

    :param source: [optional]
        The associated Source in the Astra database for this data product.

    :param star: [optional]
        The associated Star in the APOGEE DRP database for this data product.
    """
    if star is None:
        from astra.database.apogee_drpdb import Star
        if source is None:
            (source,) = data_product.sources

        # Be sure to get the record of the Star with the latest and greatest stack,
        # and latest and greatest set of Doppler values.
        star = (
            Star.select()
            .where(Star.catalogid == source.catalogid)
            .order_by(Star.created.desc())
            .first()
        )
    if star is None:
        return None

    return dict(
        telescope=star.telescope,
        mean_fiber=int(star.meanfib),
        teff=np.round(star.rv_teff, 0),
        logg=np.round(star.rv_logg, 3),
        metals=np.round(star.rv_feh, 3),
        log10vdop=utils.approximate_log10_microturbulence(star.rv_logg),
        lgvsini=1.0,
        c=0,
        n=0,
        o_mg_si_s_ca_ti=0,
    )

def initial_guess_apogeenet(data_product: DataProduct, star) -> dict:
    """
    Return an initial guess for FERRE from APOGEENet given a data product.

    :param data_product:
        The data prodcut to be analyzed with FERRE.
    """

    from astra.database.astradb import ApogeeNetOutput, TaskInputDataProducts, Task

    q = (
        ApogeeNetOutput
        .select()
        .join(Task, on=(ApogeeNetOutput.task_id == Task.id))
        .join(TaskInputDataProducts)
        .where(TaskInputDataProducts.data_product_id == data_product.id)
        .order_by(ApogeeNetOutput.snr.desc())
    )

    output = q.first()
    if output is None:
        return None
        
    teff, logg, metals = (output.teff, output.logg, output.fe_h)

    if star is None:
        print(f"WARNING: TODO: Fix this andy")
        return None
    
    return dict(
        telescope=star.telescope,
        mean_fiber=int(star.meanfib),
        teff=np.round(teff, 0),
        logg=np.round(logg, 3),
        metals=np.round(metals, 3),
        log10vdop=utils.approximate_log10_microturbulence(logg),
        lgvsini=1.0,
        c=0,
        n=0,
        o_mg_si_s_ca_ti=0,
    )


def initial_guesses(data_product: DataProduct) -> List[dict]:
    """
    Return initial guesses for FERRE given a data product.
    
    :param data_product:
        The data product containing 1D spectra for a source.
    """
    # TODO: get defaults from Star (telescope, mean_fiber, etc) in a not-so-clumsy way

    from astra.database.apogee_drpdb import Star
    (source,) = data_product.sources

    # Be sure to get the record of the Star with the latest and greatest stack,
    # and latest and greatest set of Doppler values.
    star = (
        Star.select()
        .where(Star.catalogid == source.catalogid)
        .order_by(Star.created.desc())
        .first()
    )

    try:
        int(star.meanfib)
    except:
        return []

    # TODO: Add estimates from other pipelines? Gaia?
    return [
        initial_guess_doppler(data_product, star=star),
        initial_guess_apogeenet(data_product, star=star)
    ]


def create_initial_stellar_parameter_tasks(
    input_data_products,
    header_paths: Optional[Union[List[str], Tuple[str], str]] = "$MWM_ASTRA/component_data/aspcap/synspec_dr17_marcs_header_paths.list",
    weight_path: Optional[str] = "$MWM_ASTRA/component_data/aspcap/global_mask_v02.txt",
    continuum_method: Optional[Union[Continuum, str]] = Scalar,
    continuum_kwargs: Optional[dict] = dict(method="median"),
    data_slice: Optional[List[Tuple[int]]] = [(0, 1)],
    initial_guess_callable: Optional[Callable] = None,
    **kwargs,
) -> List[Union[Task, int]]:
    """
    Create tasks that will use FERRE to estimate the stellar parameters given a data product.

    :param input_data_products:
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

    log.debug(f"Data products {type(input_data_products)}: {input_data_products}")

    # Data products.
    input_data_products = deserialize(input_data_products, DataProduct)

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
    round = lambda _, d=3: np.round(_, d).astype(float)

    # For each (data product, initial guess) permutation we need to create tasks based on suitable grids.
    task_data_products = []
    for data_product in input_data_products:
        for initial_guess in initial_guess_callable(data_product):
            if initial_guess is None:
                continue

            for header_path, meta in utils.yield_suitable_grids(
                grid_info, **initial_guess
            ):

                frozen_parameters = {}
                if meta["gd"] == "d":
                    # If it's a main-sequence grid, we freeze C and N in the initial round.
                    frozen_parameters.update(c=True, n=True)

                kwds = dict(
                    header_path=header_path,
                    weight_path=weight_path,
                    continuum_method=continuum_method,
                    continuum_kwargs=continuum_kwargs,
                    data_slice=data_slice,
                    frozen_parameters=frozen_parameters,
                    initial_parameters=dict(
                        teff=round(initial_guess["teff"], 0),
                        logg=round(initial_guess["logg"]),
                        metals=round(initial_guess["metals"]),
                        o_mg_si_s_ca_ti=round(initial_guess["o_mg_si_s_ca_ti"]),
                        lgvsini=round(initial_guess["lgvsini"]),
                        c=round(initial_guess["c"]),
                        n=round(initial_guess["n"]),
                        log10vdop=round(initial_guess["log10vdop"]),
                    ),
                )

                parameters = FERRE_DEFAULTS.copy()
                parameters.update(kwds)
                parameters.update(
                    {k: v for k, v in kwargs.items() if k in FERRE_DEFAULTS}
                )

                # Create a task.
                task = Task(
                    name=FERRE_TASK_NAME, version=__version__, parameters=parameters
                )
                task_data_products.append((task, data_product))

    with database.atomic():
        Task.bulk_create([t for t, dp in task_data_products])
        TaskInputDataProducts.insert_many([
            { "task_id": t.id, "data_product_id": dp.id } for t, dp in task_data_products
        ]).execute()

    return [t for t, dp in task_data_products]

def create_initial_stellar_parameter_task_bundles(
    input_data_products,
    header_paths: Optional[Union[List[str], Tuple[str], str]] = "$MWM_ASTRA/component_data/aspcap/synspec_dr17_marcs_header_paths.list",
    weight_path: Optional[str] = "$MWM_ASTRA/component_data/aspcap/global_mask_v02.txt",
    continuum_method: Optional[Union[Continuum, str]] = Scalar,
    continuum_kwargs: Optional[dict] = dict(method="median"),
    data_slice: Optional[List[Tuple[int]]] = [(0, 1)],
    initial_guess_callable: Optional[Callable] = None,
    **kwargs,
) -> List[Union[Bundle, int]]:

    tasks = create_initial_stellar_parameter_tasks(
        input_data_products=input_data_products,
        header_paths=header_paths,
        weight_path=weight_path,
        continuum_method=continuum_method,
        continuum_kwargs=continuum_kwargs,
        data_slice=data_slice,
        initial_guess_callable=initial_guess_callable,
        **kwargs
    )
    log.info(f"Created {len(tasks)} tasks")

    bundles = bundler(tasks)
    log.info(f"Created {len(bundles)} bundles")
    return [bundle.id for bundle in bundles]
    

def create_stellar_parameter_task_bundles(
    initial_task_bundles,
    weight_path: Optional[str] = "$MWM_ASTRA/component_data/aspcap/global_mask_v02.txt",
    continuum_method: Optional[Union[Continuum, str]] = "astra.contrib.aspcap.continuum.MedianFilter",
    continuum_kwargs: Optional[dict] = None,
    **kwargs,
):
    """
    Create FERRE tasks to estimate stellar parameters, given the best result from the
    initial round of stellar parameters.
    """

    #initial_task_bundles = deserialize(initial_task_bundles, Task)
    if isinstance(initial_task_bundles, str):
        initial_task_bundles = json.loads(initial_task_bundles)

    q = (
        Task
        .select()
        .join(TaskBundle)
        .where(TaskBundle.bundle_id.in_(initial_task_bundles))
    )
    bitmask = ParamBitMask()
    bad_grid_edge = bitmask.get_value("GRIDEDGE_WARN") | bitmask.get_value(
        "GRIDEDGE_BAD"
    )

    # Get all results per data product.
    results = {}
    for task in q:
        # TODO: Here we are assuming one data product per task, but it doesn't have to be this way.
        #       It just makes it tricky if there are many data products + results per task, as we would
        #       have to infer which result for which data product.
        (data_product,) = task.input_data_products
        parsed_header = utils.parse_header_path(
            expand_path(task.parameters["header_path"])
        )

        N_outputs = task.count_outputs()
        if N_outputs == 0:
            log.warning(f"Task {task} has no outputs!")
            continue

        results.setdefault(data_product.id, [])

        for output in task.outputs:

            penalized_log_chisq_fit = 0
            penalized_log_chisq_fit += output.log_chisq_fit

            # Penalise chi-sq in the same way they did for DR17.
            # See github.com/sdss/apogee/python/apogee/aspcap/aspcap.py#L658
            if parsed_header["spectral_type"] == "GK" and output.teff < 3900:
                log.debug(f"Increasing \chisq because spectral type GK")
                penalized_log_chisq_fit += np.log10(10)

            if output.bitmask_logg & bad_grid_edge:
                log.debug(
                    f"Increasing \chisq because bitmask on logg {output.bitmask_logg} is bad edge ({bad_grid_edge})"
                )
                penalized_log_chisq_fit += np.log10(5)

            if output.bitmask_teff & bad_grid_edge:
                log.debug(
                    f"Increasing \chisq because bitmask on teff {output.bitmask_teff} is bad edge ({bad_grid_edge})"
                )
                penalized_log_chisq_fit += np.log10(5)

            result = (penalized_log_chisq_fit, task, output)
            results[data_product.id].append(result)

    # Let's update these tasks with their penalized values!
    with database.atomic():
        for _, task_results in results.items():
            for (penalized_log_chisq_fit, task, output) in task_results:
                log.info(
                    f"Setting penalized log chisq = {penalized_log_chisq_fit} for task {task} and output {output}"
                )
                rows = (
                    FerreOutput.update(penalized_log_chisq_fit=penalized_log_chisq_fit)
                    .where(FerreOutput.output == output)
                    .execute()
                )
                assert rows == 1

    # Order all results from (best, recent) to (worst, older).
    # The initial round of stellar parameters should only have one result per task.
    # If there are multiple results per task, it means the task has been re-executed a few times.
    # Here we will take the lowest log_chisq_fit, highest task (more recent), and highest output (more recent).
    results = {
        dp: sorted(values, key=lambda row: (row[0], -row[1].id, -row[2].output.id))
        for dp, values in results.items()
    }
    for dp, values in results.items():
        log.info(f"For data product {dp}:")
        for i, (log_chisq_fit, task, output) in enumerate(values):
            log.info(
                f"\t{i:.0f}: \chi^2 = {log_chisq_fit:.3f} for task {task} and output {output}"
            )

    if continuum_method is not None and not isinstance(continuum_method, str):
        continuum_method = serialize_executable(continuum_method)

    task_data_products = []
    for data_product_id, (result, *_) in results.items():
        log_chisq_fit, task, output = result

        # For the normalization we will do a median filter correction using the previous result.
        if continuum_method is not None:
            _continuum_kwargs = (continuum_kwargs or {}).copy()
            _continuum_kwargs.update(upstream_task_id=task.id)
        else:
            _continuum_kwargs = FERRE_DEFAULTS["continuum_kwargs"]

        parameters = FERRE_DEFAULTS.copy()
        parameters.update(
            header_path=task.parameters["header_path"],
            weight_path=weight_path,
            continuum_method=continuum_method,
            continuum_kwargs=_continuum_kwargs,
            initial_parameters=dict(
                teff=output.teff,
                logg=output.logg,
                metals=output.metals,
                o_mg_si_s_ca_ti=output.o_mg_si_s_ca_ti,
                lgvsini=output.lgvsini,
                c=output.c,
                n=output.n,
                log10vdop=output.log10vdop,
            ),
        )
        parameters.update({k: v for k, v in kwargs.items() if k in FERRE_DEFAULTS})

        task = Task(
            name=FERRE_TASK_NAME, version=__version__, parameters=parameters
        )
        task_data_products.append((task, data_product_id))

    with database.atomic():
        Task.bulk_create([t for t, _ in task_data_products])
        TaskInputDataProducts.insert_many([
            { "task_id": t.id, "data_product_id": dp_id } for t, dp_id in task_data_products
        ]).execute()

    tasks = [t for t, _ in task_data_products]

    # Create bundles
    log.info(f"Created {len(tasks)} tasks")

    bundles = bundler(tasks)
    log.info(f"Created {len(bundles)} bundles")
    return [bundle.id for bundle in bundles]



def get_element(weight_path):
    return os.path.basename(weight_path)[:-5]


def create_abundance_tasks(
    stellar_parameter_bundles,
    weight_paths: str = "$MWM_ASTRA/component_data/aspcap/element_masks.list",
    as_primary_keys: bool = False,
    **kwargs,
) -> List[Task]:
    """
    Create FERRE tasks to estimate chemical abundances, given the stellar parameters determined from previous tasks.
    """

    # Load the weight paths.
    with open(expand_path(weight_paths), "r") as fp:
        weight_paths = list(map(str.strip, fp.readlines()))

    all_headers = {}
    abundance_keywords = {}

    if isinstance(stellar_parameter_bundles, str):
        stellar_parameter_bundles = json.loads(stellar_parameter_bundles)
        
    q = (
        Task
        .select()
        .distinct(Task)
        .join(TaskBundle)
        .switch(Task)
        .join(FerreOutput, on=(Task.id == FerreOutput.task_id)) # Restrict to things that have outputs
        .where(TaskBundle.bundle_id.in_(stellar_parameter_bundles))
    )

    tasks = []
    for task in q:
        header_path = task.parameters["header_path"]
        if header_path not in abundance_keywords:
            abundance_keywords[header_path] = {}
            headers, *segment_headers = read_ferre_headers(expand_path(header_path))
            all_headers[header_path] = (headers, *segment_headers)
            for weight_path in weight_paths:
                element = get_element(weight_path)
                abundance_keywords[header_path][element] = utils.get_abundance_keywords(
                    element, headers["LABEL"]
                )

        # Set initial parameters to the stellar parameters determined from the previous task.
        initial_parameters = []
        for output in task.outputs:
            initial_parameters.append(
                dict(
                    teff=output.teff,
                    logg=output.logg,
                    metals=output.metals,
                    o_mg_si_s_ca_ti=output.o_mg_si_s_ca_ti,
                    lgvsini=output.lgvsini,
                    c=output.c,
                    n=output.n,
                    log10vdop=output.log10vdop,
                )
            )
        log.debug(
            f"From task {task} with {list(task.input_data_products)} input {len(initial_parameters)} stellar parameters"
        )

        for weight_path in weight_paths:

            element = get_element(weight_path)

            frozen_parameters, ferre_abundance_kwds = abundance_keywords[header_path][
                element
            ]

            parameters = FERRE_DEFAULTS.copy()
            parameters.update(task.parameters)
            parameters.update(
                weight_path=weight_path,
                initial_parameters=initial_parameters,
                frozen_parameters=frozen_parameters,
            )
            parameters.update({k: v for k, v in kwargs.items() if k in FERRE_DEFAULTS})
            parameters["ferre_kwds"] = parameters["ferre_kwds"] or dict()
            parameters["ferre_kwds"].update(ferre_abundance_kwds)

            # Check to see if all parameters are going to be frozen.
            n_of_dim = all_headers[header_path][0]["N_OF_DIM"]
            n_frozen_dim = sum(frozen_parameters.values())
            n_free_dim = n_of_dim - n_frozen_dim
            if n_free_dim == 0:
                log.warning(
                    f"Not creating task {FERRE_TASK_NAME} with weight path {weight_path} from task {task} "
                    f"because all parameters are frozen (n_of_dim: {n_of_dim}, n_frozen: {n_frozen_dim}):\n{parameters}"
                )
                continue

            print(f"TODO: insert tasks in bulk instead")
            abundance_task = Task.create(
                name=FERRE_TASK_NAME, version=__version__, parameters=parameters
            )
            log.debug(f"Created {abundance_task} from it")
            for data_product in task.input_data_products:
                TaskInputDataProducts.create(
                    task=abundance_task, data_product=data_product
                )

            tasks.append(abundance_task)

    if as_primary_keys:
        return [task.id for task in tasks]
    return tasks


class Aspcap(TaskInstance):

    stellar_parameter_task_id = Parameter()
    chemical_abundance_task_ids = TupleParameter()

    def execute(self):
        hdu, extname = (3, "APOGEE/APO") # TODO: get extname in future

        ferre_parameter_names = ("teff", "logg", "metals", "o_mg_si_s_ca_ti", "log10vdop", "lgvsini", "c", "n")

        results = []
        for task, input_data_products, parameters in self.iterable():
            log.info(f"Executing task {task}")
            stellar_parameter_task = Task.get(int(task.parameters["stellar_parameter_task_id"]))

            data_product, = stellar_parameter_task.input_data_products
            TaskInputDataProducts.create(task=task, data_product=data_product)
            
            task_results = []
            hdu_results = {}
            if stellar_parameter_task.count_outputs() == 0:
                # Nothing we can do about that.
                log.warning(f"Skipping task {task} because stellar parameter task {stellar_parameter_task} has no outputs")
                continue

            for output in stellar_parameter_task.outputs:
                result = {}
                for key in ferre_parameter_names:
                    result[key] = getattr(output, key) or np.nan
                    result[f"e_{key}"] = getattr(output, f"e_{key}") or np.nan
                    result[f"bitmask_{key}"] = getattr(output, f"bitmask_{key}") or 0
                
                result.update(OrderedDict([
                    ("snr", output.snr),
                    ("log_chisq_fit", output.log_chisq_fit),
                    ("log_snr_sq", output.log_snr_sq),
                    #("frac_phot_data_points", output.frac_phot_data_points),
                    ("source_id", output.source_id),
                    ("parent_data_product_id", output.parent_data_product_id),
                ]))
                task_results.append(result)

            # Load the spectra from the stellar parameter task.
            with fits.open(stellar_parameter_task.output_data_products[0].path) as image:
                spectra = OrderedDict([
                    ("model_flux", image[hdu].data["MODEL_FLUX"]),
                    ("continuum", image[hdu].data["CONTINUUM"]),
                ])

            elements = []
            for abundance_task in Task.select().where(Task.id << tuple(map(int, task.parameters["chemical_abundance_task_ids"]))):
                if abundance_task.count_outputs() == 0:
                    log.warning(
                        f"No outputs for abundance task {abundance_task} on stellar parameter task {stellar_parameter_task}. "
                        "Skipping!"
                    )
                    continue

                # get the element
                element = get_element(abundance_task.parameters["weight_path"])

                # Check which parameters are frozen.
                initial = abundance_task.parameters["initial_parameters"]
                initial = initial[0] if isinstance(initial, list) else initial
                initial = sanitise([k for k, v in initial.items() if v is not None])
                frozen = sanitise(
                    [
                        k
                        for k, v in abundance_task.parameters[
                            "frozen_parameters"
                        ].items()
                        if v
                    ]
                )
                parameter_name = set(initial).difference(frozen)
                if len(parameter_name) > 1:
                    log.warning(f"Parameter names thawed: {parameter_name}")
                    parameter_name = parameter_name.difference({"lgvsini"})

                if len(parameter_name) == 0:
                    log.warning(f"No free parameters for {abundance_task}")
                    log.debug(
                        f"initial: {initial}: {abundance_task.parameters['initial_parameters']}"
                    )
                    log.debug(
                        f"frozen: {frozen}: {abundance_task.parameters['frozen_parameters']}"
                    )
                    continue

                assert len(parameter_name) == 1
                (parameter_name,) = parameter_name

                key = f"{element.lower()}_h"
                elements.append(key)
                for i, output in enumerate(abundance_task.outputs):
                    task_results[i][key] = getattr(output, parameter_name)
                    task_results[i][f"e_{key}"] = getattr(output, f"e_{parameter_name}")
                    task_results[i][f"bitmask_{key}"] = getattr(
                        output, f"bitmask_{parameter_name}"
                    )
                    task_results[i][f"log_chisq_fit_{key}"] = output.log_chisq_fit
                
                with fits.open(abundance_task.output_data_products[0].path) as image:
                    spectra[f"model_flux_{key}"] = image[hdu].data["MODEL_FLUX"]
                    spectra[f"continuum_{key}"] = image[hdu].data["CONTINUUM"]

            unsorted_hdu_result = list_to_dict(task_results)

            hdu_result = OrderedDict()
            for key in ferre_parameter_names:
                hdu_result[key] = unsorted_hdu_result[key]
                hdu_result[f"e_{key}"] = unsorted_hdu_result[f"e_{key}"]
            
            for key in elements:
                hdu_result[key] = unsorted_hdu_result[key]
                hdu_result[f"e_{key}"] = unsorted_hdu_result[f"e_{key}"]
            
            # bitmasks
            for key in ferre_parameter_names:
                hdu_result[f"bitmask_{key}"] = unsorted_hdu_result[f"bitmask_{key}"]
            for key in elements:
                hdu_result[f"bitmask_{key}"] = unsorted_hdu_result[f"bitmask_{key}"]
            
            # summary statistics
            for key in ("snr", "log_chisq_fit"):
                hdu_result[key] = unsorted_hdu_result[key]
            for key in elements:
                hdu_result[f"log_chisq_fit_{key}"] = unsorted_hdu_result[f"log_chisq_fit_{key}"]
            
            # identifiers
            for key in ("source_id", "parent_data_product_id"):
                hdu_result[key] = unsorted_hdu_result[key]
            
            # spectra
            for key in ("model_flux", "continuum"):
                hdu_result[key] = spectra[key]
            
            for key in elements:
                hdu_result[f"model_flux_{key}"] = spectra[f"model_flux_{key}"]
                hdu_result[f"continuum_{key}"] = spectra[f"continuum_{key}"]

            # create
            header_groups = [
                ("TEFF", "STELLAR PARAMETERS"),
            ]
            if len(elements):
                header_groups.append((elements[0].upper(), "CHEMICAL ABUNDANCES"))
            header_groups.extend([
                ("BITMASK_TEFF", "BITMASKS"),
                ("SNR", "SUMMARY STATISTICS"),
                ("SOURCE_ID", "IDENTIFIERS"),
                ("MODEL_FLUX", "MODEL SPECTRA"),
            ])
            odp = create_pipeline_product(
                task, 
                data_product, 
                { extname: hdu_result }, 
                header_groups={
                    extname: header_groups
                }
            )
            with database.atomic():
                task.create_or_update_outputs(AspcapOutput, task_results)
            
'''
class OldAspcap(TaskInstance):

    stellar_parameter_task_ids = TupleParameter("stellar_parameter_task_ids")
    abundance_task_ids = TupleParameter("abundance_task_ids")

    def execute(self):

        results = []

        for task, input_data_products, parameters in self.iterable():

            stellar_parameter_task = Task.get_by_id(
                int(parameters["stellar_parameter_task_ids"])
            )
            log.debug(
                f"ASPCAP task {task} got stellar parameter task {stellar_parameter_task}"
            )
            for data_product in stellar_parameter_task.input_data_products:
                log.debug(f"Assigned data product {data_product} to ASPCAP task {task}")
                TaskInputDataProducts.create(task=task, data_product=data_product)

            abundance_tasks = deserialize(parameters["abundance_task_ids"], Task)
            log.debug(f"ASPCAP task {task} got abundance tasks {abundance_tasks}")

            task_results = []
            for output in stellar_parameter_task.outputs:
                result = dict(
                    snr=output.snr,
                    log_chisq_fit=output.log_chisq_fit,
                    log_snr_sq=output.log_snr_sq,
                    frac_phot_data_points=output.frac_phot_data_points,
                    meta=output.meta,
                )
                for key in (
                    "teff",
                    "logg",
                    "metals",
                    "o_mg_si_s_ca_ti",
                    "log10vdop",
                    "lgvsini",
                    "c",
                    "n",
                ):
                    result[key] = getattr(output, key)
                    result[f"u_{key}"] = getattr(output, f"u_{key}")
                    result[f"bitmask_{key}"] = getattr(output, f"bitmask_{key}")

                task_results.append(result)

            for abundance_task in abundance_tasks:

                if abundance_task.count_outputs() == 0:
                    log.warning(
                        f"No outputs for abundance task {abundance_task} on stellar parameter task {stellar_parameter_task}. "
                        "Skipping!"
                    )
                    continue

                # get the element
                element = get_element(abundance_task.parameters["weight_path"])

                # Check which parameters are frozen.
                initial = abundance_task.parameters["initial_parameters"]
                initial = initial[0] if isinstance(initial, list) else initial
                initial = sanitise([k for k, v in initial.items() if v is not None])
                frozen = sanitise(
                    [
                        k
                        for k, v in abundance_task.parameters[
                            "frozen_parameters"
                        ].items()
                        if v
                    ]
                )
                parameter_name = set(initial).difference(frozen)
                if len(parameter_name) > 1:
                    log.warning(f"Parameter names thawed: {parameter_name}")
                    parameter_name = parameter_name.difference({"lgvsini"})

                if len(parameter_name) == 0:
                    log.warning(f"No free parameters for {abundance_task}")
                    log.debug(
                        f"initial: {initial}: {abundance_task.parameters['initial_parameters']}"
                    )
                    log.debug(
                        f"frozen: {frozen}: {abundance_task.parameters['frozen_parameters']}"
                    )
                    continue

                assert len(parameter_name) == 1
                (parameter_name,) = parameter_name

                key = f"{element.lower()}_h"
                for i, output in enumerate(abundance_task.outputs):
                    task_results[i][key] = getattr(output, parameter_name)
                    task_results[i][f"u_{key}"] = getattr(output, f"u_{parameter_name}")
                    task_results[i][f"bitmask_{key}"] = getattr(
                        output, f"bitmask_{parameter_name}"
                    )
                    task_results[i][f"log_chisq_fit_{key}"] = output.log_chisq_fit

            for result in task_results:
                output = Output.create()
                TaskOutput.create(task=task, output=output)
                AspcapOutput.create(task=task, output=output, **result)

            results.append(task_results)
        return results
'''

def create_and_execute_summary_tasks(
    stellar_parameter_bundles,
    chemical_abundance_bundles,
):
    """
    Create a row in the AspcapOutput database table for each input data product,
    given the stellar parameter tasks and the abundance tasks.
    """

    if isinstance(stellar_parameter_bundles, str):
        stellar_parameter_bundles = json.loads(stellar_parameter_bundles)
    if isinstance(chemical_abundance_bundles, str):
        chemical_abundance_bundles = json.loads(chemical_abundance_bundles)

    # Join by input data product.
    #for task in Task.select().join(TaskBundle).where(TaskBundle.bundle_id.in_(stellar_parameter_bundles)):
    #    (data_product_id,) = task.input_data_products
    #    grouped[data_product_id] = [task.id]
    q = (
        Task
        .select(Task.id, DataProduct.id)
        .join(TaskBundle)
        .switch(Task)
        .join(TaskInputDataProducts)
        .join(DataProduct)
        .where(TaskBundle.bundle_id.in_(stellar_parameter_bundles))
        .tuples()
    )
    grouped = { data_product_id: [task_id] for task_id, data_product_id in q }

    q = (
        Task
        .select(Task.id, DataProduct.id)
        .join(TaskBundle)
        .switch(Task)
        .join(TaskInputDataProducts)
        .join(DataProduct)
        .where(TaskBundle.bundle_id.in_(chemical_abundance_bundles))
        .tuples()
    )
    for task_id, data_product_id in q:
        grouped[data_product_id].append(task_id)
    #for task in Task.select().join(TaskBundle).where(TaskBundle.bundle_id.in_(chemical_abundance_bundles)):
    #    (data_product_id,) = task.input_data_products
    #    grouped[data_product_id].append(task.id)

    # Create and execute tasks.
    for data_product_id, (
        stellar_parameter_task_id,
        *chemical_abundance_task_ids,
    ) in grouped.items():
        log.debug(
            f"Checking data product {data_product_id} with stellar parameter task {stellar_parameter_task_id} and abundance tasks {chemical_abundance_task_ids}"
        )
        # Check to see if it already exists?
        # TODO: revisit this, it is slow and dumb. doing it just to make sure things are idempotent
        q = (
            Task
            .select()
            .where(
                (Task.name == 'astra.contrib.aspcap.base.Aspcap')
            &   (Task.version == __version__)
            &   (Task.parameters["stellar_parameter_task_id"] == f"{stellar_parameter_task_id}")
            )
        )
        # TODO: check by chemical abundances as well?
        existing_task = q.first()
        if existing_task is None:
            log.info(f"Creating task")
            Aspcap(
                stellar_parameter_task_id=stellar_parameter_task_id,
                chemical_abundance_task_ids=chemical_abundance_task_ids,
            ).execute()
        else:
            log.info(f"Skipping {data_product_id}, {stellar_parameter_task_id}")
        #else:
        #    existing_task.instance().execute()

    return None
