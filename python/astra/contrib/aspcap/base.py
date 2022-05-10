import os
from re import findall
import numpy as np
import json
from astra.contrib.aspcap import continuum, utils
from astra.tools.continuum.base import NormalizationBase
from typing import Union, List, Tuple, Optional, Callable
from tqdm import tqdm

from astra import log, __version__
from astra.base import ExecutableTask
from astra.utils import (deserialize, expand_path, serialize_executable)
from astra.database.astradb import (FerreOutput, Source, Task, DataProduct, TaskInputDataProducts)
from astra.database.apogee_drpdb import Star
from astra.contrib.ferre.base import Ferre
from astra.contrib.ferre.utils import (read_ferre_headers, sanitise)
from astra.contrib.ferre.bitmask import ParamBitMask

FERRE_TASK_NAME = serialize_executable(Ferre)
FERRE_DEFAULTS = Ferre.get_defaults()

def initial_guess_doppler(
        data_product: DataProduct, 
        source: Optional[Source] = None, 
        star: Optional[Star] = None
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
        if source is None:
            source, = data_product.sources

        # Be sure to get the record of the Star with the latest and greatest stack,
        # and latest and greatest set of Doppler values.
        star = (
            Star.select()
                .where(Star.catalogid == source.catalogid)
                .order_by(Star.created.desc())
                .first()
        )

    return dict(
        telescope=data_product.kwargs["telescope"],
        mean_fiber=int(star.meanfib),
        teff=np.round(star.rv_teff, 0),
        logg=np.round(star.rv_logg, 3),
        metals=np.round(star.rv_feh, 3),
        log10vdop=utils.approximate_log10_microturbulence(star.rv_logg),
        lgvsini=1.0,
        c=0,
        n=0,
        o_mg_si_s_ca_ti=0
    )


def initial_guesses(data_product: DataProduct) -> List[dict]:
    """ Return initial guesses for FERRE given a data product. """
    return [initial_guess_doppler(data_product)]


def create_initial_stellar_parameter_tasks(
        input_data_products,
        header_paths: Union[List[str], Tuple[str], str],
        weight_path: Optional[str] = "$MWM_ASTRA/component_data/aspcap/global_mask_v02.txt",
        normalization_method: Optional[Union[NormalizationBase, str]] = continuum.MedianNormalizationWithErrorInflation,
        slice_args: Optional[List[Tuple[int]]] = [(0, 1)],
        initial_guess_callable: Optional[Callable] = None,
        as_primary_keys: bool = False,
        **kwargs
    ) -> List[Task]:
    """
    Create tasks that will use FERRE to estimate the stellar parameters given the stacked spectrum in an ApStar data product.
    
    :param input_data_products:
        The input (ApStar) data products, or primary keys for those data products.

    :param header_paths:
        A list of FERRE header path files, or a path to a file that has one FERRE header path per line.

    :param weight_path: [optional]
        The weights path to supply to FERRE. By default this is set to the global mask used by SDSS.
    
    :param normalization_method: [optional]
        The method to use for continuum normalization before FERRE is executed. By default this is set to 

    :param slice_args: [optional]
        Slice the input spectra and only analyze those rows that meet the slice. Because this is the initial
        round of stellar parameter determination, by default we only take the highest S/N spectrum (i.e., the
        first spectrum in each ApStar data product).
    
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

    :param as_primary_keys: [optional]
        Return a list of primary keys instead of tasks. 
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

    if normalization_method is not None:
        normalization_method = serialize_executable(normalization_method)

    grid_info = utils.parse_grid_information(header_paths)
    
    if initial_guess_callable is None:
        initial_guess_callable = initial_guesses

    # Round the initial guesses to something sensible.
    round = lambda _, d=3: np.round(_, d).astype(float)
    
    # For each (data product, initial guess) permutation we need to create tasks based on suitable grids.
    tasks = []
    for data_product in input_data_products:
        for initial_guess in initial_guess_callable(data_product):
            if initial_guess is None: continue

            for header_path, meta in utils.yield_suitable_grids(grid_info, **initial_guess):

                frozen_parameters = {}
                if meta["gd"] == "d":
                    # If it's a main-sequence grid, we freeze C and N in the initial round.
                    frozen_parameters.update(c=True, n=True)
            
                kwds = dict(
                    header_path=header_path,
                    weight_path=weight_path,
                    normalization_method=normalization_method,
                    slice_args=slice_args,
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
                parameters.update({ k: v for k, v in kwargs.items() if k in FERRE_DEFAULTS })
                
                # Create a task.
                task = Task.create(
                    name=FERRE_TASK_NAME,
                    version=__version__,
                    parameters=parameters
                )
                TaskInputDataProducts.create(task=task, data_product=data_product)
                tasks.append(task)
    
    if as_primary_keys:
        return [task.id for task in tasks] 
    return tasks
    


def create_stellar_parameter_tasks_from_best_initial_tasks(
        initial_tasks,
        weight_path: Optional[str] = "$MWM_ASTRA/component_data/aspcap/global_mask_v02.txt",
        normalization_method: Optional[Union[NormalizationBase, str]] = continuum.MedianFilterNormalizationWithErrorInflation,
        normalization_kwds: Optional[dict] = None,
        as_primary_keys: bool = False,
        **kwargs
    ) -> List[Task]:
    """
    Create FERRE tasks to estimate stellar parameters, given the best result from the 
    initial round of stellar parameters.
    """

    initial_tasks = deserialize(initial_tasks, Task)

    bitmask = ParamBitMask()
    bad_grid_edge = (bitmask.get_value("GRIDEDGE_WARN") | bitmask.get_value("GRIDEDGE_BAD"))

    # Get all results per data product.
    results = {}
    for task in initial_tasks:
        # TODO: Here we are assuming one data product per task, but it doesn't have to be this way.
        #       It just makes it tricky if there are many data products + results per task, as we would
        #       have to infer which result for which data product.
        data_product, = task.input_data_products
        parsed_header = utils.parse_header_path(expand_path(task.parameters["header_path"]))

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
                log.debug(f"Increasing \chisq because bitmask on logg {output.bitmask_logg} is bad edge ({bad_grid_edge})")
                penalized_log_chisq_fit += np.log10(5)
                
            if output.bitmask_teff & bad_grid_edge:
                log.debug(f"Increasing \chisq because bitmask on teff {output.bitmask_teff} is bad edge ({bad_grid_edge})")
                penalized_log_chisq_fit += np.log10(5)
            
            result = (penalized_log_chisq_fit, task, output)
            results[data_product.id].append(result)

    # Let's update these tasks with their penalized values!
    with database.atomic():
        for _, task_results in results.items():
            for (penalized_log_chisq_fit, task, output) in task_results:
                log.info(f"Setting penalized log chisq = {penalized_log_chisq_fit} for task {task} and output {output}")
                rows = (
                    FerreOutput.update(penalized_log_chisq_fit=penalized_log_chisq_fit)
                               .where(FerreOutput.output==output)
                               .execute()
                )
                assert rows == 1

    # Order all results from (best, recent) to (worst, older).
    # The initial round of stellar parameters should only have one result per task.
    # If there are multiple results per task, it means the task has been re-executed a few times.
    # Here we will take the lowest log_chisq_fit, highest task (more recent), and highest output (more recent).
    results = { dp: sorted(values, key=lambda row: (row[0], -row[1].id, -row[2].output.id)) for dp, values in results.items() }
    for dp, values in results.items():
        log.info(f"For data product {dp}:")
        for i, (log_chisq_fit, task, output) in enumerate(values):
            log.info(f"\t{i:.0f}: \chi^2 = {log_chisq_fit:.3f} for task {task} and output {output}")

    if normalization_method is not None:
        normalization_method = serialize_executable(normalization_method)

    tasks = []
    for data_product_id, (result, *_) in results.items():
        log_chisq_fit, task, output = result
        
        # For the normalization we will do a median filter correction using the previous result.
        if normalization_method is not None:
            _normalization_kwds = (normalization_kwds or {}).copy()
            _normalization_kwds.update(median_filter_from_task=task.id)
        else:
            _normalization_kwds = FERRE_DEFAULTS["normalization_kwds"]

        parameters = FERRE_DEFAULTS.copy()
        parameters.update(
            header_path=task.parameters["header_path"],
            weight_path=weight_path,
            normalization_method=normalization_method,
            normalization_kwds=_normalization_kwds,
            initial_parameters=dict(
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
        parameters.update({ k: v for k, v in kwargs.items() if k in FERRE_DEFAULTS })

        task = Task.create(
            name=FERRE_TASK_NAME,
            version=__version__,
            parameters=parameters
        )
        TaskInputDataProducts.create(task=task, data_product_id=data_product_id)
        tasks.append(task)
    
    if as_primary_keys:
        return [task.id for task in tasks]
    return tasks


def get_element(weight_path):
    return os.path.basename(weight_path)[:-5]


def create_abundance_tasks(
        stellar_parameter_tasks,
        weight_paths: str = "$MWM_ASTRA/component_data/aspcap/element_masks.list",
        as_primary_keys: bool = False,
        **kwargs
    ) -> List[Task]:
    """
    Create FERRE tasks to estimate chemical abundances, given the stellar parameters determined from previous tasks.
    """
    
    stellar_parameter_tasks = deserialize(stellar_parameter_tasks, Task)

    # Load the weight paths.
    with open(expand_path(weight_paths), "r") as fp:
        weight_paths = list(map(str.strip, fp.readlines()))
    
    all_headers = {}
    abundance_keywords = {}

    tasks = []
    for task in stellar_parameter_tasks:

        header_path = task.parameters["header_path"]
        if header_path not in abundance_keywords:
            abundance_keywords[header_path] = {}
            headers, *segment_headers = read_ferre_headers(expand_path(header_path))
            all_headers[header_path] = (headers, *segment_headers)
            for weight_path in weight_paths:
                element = get_element(weight_path)
                abundance_keywords[header_path][element] = utils.get_abundance_keywords(element, headers["LABEL"])

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
        log.debug(f"From task {task} with {list(task.input_data_products)} input {len(initial_parameters)} stellar parameters")

        for weight_path in weight_paths:

            element = get_element(weight_path)

            frozen_parameters, ferre_abundance_kwds = abundance_keywords[header_path][element]
            
            parameters = FERRE_DEFAULTS.copy()
            parameters.update(task.parameters)
            parameters.update(
                weight_path=weight_path,
                initial_parameters=initial_parameters,
                frozen_parameters=frozen_parameters,
            )
            parameters.update({ k: v for k, v in kwargs.items() if k in FERRE_DEFAULTS })
            parameters["ferre_kwds"] = (parameters["ferre_kwds"] or dict())
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

            abundance_task = Task.create(
                name=FERRE_TASK_NAME,
                version=__version__,
                parameters=parameters
            )
            log.debug(f"Created {abundance_task} from it")
            for data_product in task.input_data_products:
                TaskInputDataProducts.create(task=abundance_task, data_product=data_product)

            tasks.append(abundance_task)

    if as_primary_keys:
        return [task.id for task in tasks]
    return tasks




from astra.database.astradb import database, Task, TaskInputDataProducts, AspcapOutput, Output, TaskOutput
from astra.base import ExecutableTask, TupleParameter

class Aspcap(ExecutableTask):

    stellar_parameter_task_ids = TupleParameter("stellar_parameter_task_ids")
    abundance_task_ids = TupleParameter("abundance_task_ids")

    def execute(self):

        results = []

        for task, input_data_products, parameters in self.iterable():

            stellar_parameter_task = Task.get_by_id(int(parameters["stellar_parameter_task_ids"]))
            log.debug(f"ASPCAP task {task} got stellar parameter task {stellar_parameter_task}")
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
                    meta=output.meta
                )
                for key in ("teff", "logg", "metals", "o_mg_si_s_ca_ti", "log10vdop", "lgvsini", "c", "n"):
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
                frozen = sanitise([k for k, v in abundance_task.parameters["frozen_parameters"].items() if v])
                parameter_name = set(initial).difference(frozen)
                if len(parameter_name) > 1:
                    log.warning(f"Parameter names thawed: {parameter_name}")
                    parameter_name = parameter_name.difference({"lgvsini"})
                
                if len(parameter_name) == 0:
                    log.warning(f"No free parameters for {abundance_task}")
                    log.debug(f"initial: {initial}: {abundance_task.parameters['initial_parameters']}")
                    log.debug(f"frozen: {frozen}: {abundance_task.parameters['frozen_parameters']}")
                    continue

                assert len(parameter_name) == 1
                parameter_name, = parameter_name

                key = f"{element.lower()}_h"
                for i, output in enumerate(abundance_task.outputs):
                    task_results[i][key] = getattr(output, parameter_name)
                    task_results[i][f"u_{key}"] = getattr(output, f"u_{parameter_name}")
                    task_results[i][f"bitmask_{key}"] = getattr(output, f"bitmask_{parameter_name}")
                    task_results[i][f"log_chisq_fit_{key}"] = output.log_chisq_fit
            
            for result in task_results:
                output = Output.create()
                TaskOutput.create(task=task, output=output)
                AspcapOutput.create(
                    task=task,
                    output=output,
                    **result
                )

            results.append(task_results)
        return results                




def create_and_execute_summary_tasks(
        stellar_parameter_tasks,
        abundance_tasks,   
    ):
    """
    Create a row in the AspcapOutput database table for each input data product,
    given the stellar parameter tasks and the abundance tasks.
    """

    stellar_parameter_tasks = deserialize(stellar_parameter_tasks, Task)
    abundance_tasks = deserialize(abundance_tasks, Task)
    
    # Join by input data product.
    grouped = {}
    for task in stellar_parameter_tasks:
        data_product_id, = task.input_data_products
        grouped[data_product_id] = [task.id]
    
    for task in abundance_tasks:
        data_product_id, = task.input_data_products
        grouped[data_product_id].append(task.id)

    # Create and execute tasks.
    for data_product_id, (stellar_parameter_task_ids, *abundance_task_ids) in grouped.items():
        log.debug(f"Creating Aspcap summary task for data product {data_product_id} with stellar parameter task {stellar_parameter_task_ids} and abundance tasks {abundance_task_ids}")
        task = Aspcap(
            stellar_parameter_task_ids=stellar_parameter_task_ids,
            abundance_task_ids=abundance_task_ids
        )
        task.execute()
    
    return None