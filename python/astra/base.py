import inspect
import numpy as np
import os
from itertools import repeat, zip_longest
from decorator import decorator
from time import time
from typing import Union, Dict, get_origin, get_args
from peewee import IntegrityError, IntegerField, BooleanField, TextField, FloatField
from playhouse.postgres_ext import ArrayField
from playhouse.postgres_ext import BinaryJSONField as JSONField

from astra.utils import log, flatten
from astra.database.astradb import Task, BaseModel, BaseTaskOutput, database


_DATA_PRODUCT_KEY = "data_product"
_SOURCE_KEY = "source"


@decorator
def task_decorator(
    function, 
    checkpoint_frequency=300, 
    execution_context=None, *args, 
    **kwargs
):
    """
    A decorator for functions that serve as Astra tasks.

    :param function:
        The callable to decorate.

    :param checkpoint_frequency: [optional]
        The number of seconds to wait before saving the results to the database.
    
    :param execution_context: [optional]
        The execution context for the task. If None, then the task is executed immediately.
    """

    # If execution context is None, then just execute here.
    assert execution_context is None, "No execution context available yet."


    '''
    # These are the different execution modes:

    - One-to-many mapping:          data_product: DataProduct -> Iterable[Output]
    - Iterable one-to-one mapping:  data_product: Iterable[DataProduct] -> Iterable[Output]

    If it's a one-to-many mapping, then `params` is the same for all outputs.
    If it's not, then we need to see what other arguments are `Iterable`, which we will
    assume is the same list as `data_products`.
    '''

    params = _parse_arguments(function, *args, **kwargs)

    is_generator_function = inspect.isgeneratorfunction(function)

    now = time()
    
    if is_generator_function:
        results, times, last_checkpoint = ([], [now], now)
        try:
            for i, r in enumerate(function(*args, **kwargs)):
                times.append(time())
                results.append(r)

                # If we are at the number of save points, save the results.
                if (times[-1] - last_checkpoint) > checkpoint_frequency:

                    _write_database_outputs(params, results, np.diff(times), function)

                    now = time()
                    results, times, last_checkpoint = ([], [now], now)

        except StopIteration:
            None

        except:
            # TODO: Log it somewhere so we can recover it in interactive mode.
            log.exception(f"Exception raise in task {function.__name__}")
            raise

        else:
            time_elapsed = np.diff(times)
    else:
        t_init = time()
        try:
            results = function(*args, **kwargs)
        except:
            # TODO: Log it somewhere so we can recover it in interactive mode.
            log.exception(f"Exception raise in task {function.__name__}")            
            raise
        
        flat_results = flatten(results)
        N = len(flat_results)
        time_elapsed = repeat((time() - t_init) / N)

    _write_database_outputs(params, results, time_elapsed, function)

    # TODO: a registry of tasks and their output tables so that we can instantiate them from the database.
    if is_generator_function:
        yield from results
    else:
        return results

task = task_decorator


def _write_database_outputs(params, results, time_elapsed, function, time_bundle=0):

    for r, p, te in zip(results, params, time_elapsed):
        N = len(r) if isinstance(r, (tuple, list)) else 1

        # To allow for one-to-many mappings, we need to update the parameters for each result.
        for result in flatten(r):
            # TODO: sanity check that results and params dont contain conflicting information?
            common_keys = set(p).intersection(result.__data__)
            for k in common_keys:
                expected_from_params = p[k]
                actual_from_result = result.__data__[k]

                if expected_from_params != actual_from_result and k == "data_product":
                    try:
                        expected_from_params = expected_from_params.id
                    except:
                        None
                    try:
                        actual_from_result = actual_from_result.id
                    except:
                        None
                    
                if expected_from_params != actual_from_result:
                    log.warning(f"Overwriting {k}={actual_from_result} instead of {expected_from_params} for {function.__name__}.")
                    raise a

            result.__data__.update(**p)
            result.__data__.setdefault("time_elapsed", te / N)
            result.__data__.setdefault("time_bundle", time_bundle)

    # TODO: make this more explicit? ASTRA_WRITE_OUTPUTS?
    if os.getenv("ASTRA_ENVIRONMENT", None) in ("dev", "develop", "staging"):
        log.warning(f"Not writing database outputs because ASTRA_ENVIRONMENT is {os.getenv('ASTRA_ENVIRONMENT')}.")
    
    else:
        # TODO: If no Astra database, just return the results.
        model = _get_or_create_database_table_for_task(function)
        items = []
        needs_task_at_index = []
        for i, item in enumerate(filter(lambda x: isinstance(x, BaseModel), flatten(results))):
            if hasattr(item, "task_id") and isinstance(item.task_id, Task) and item.task_id.id is None:
                needs_task_at_index.append(i)
            items.append(item)

        if needs_task_at_index:
            print(f"adding {len(needs_task_at_index)} tasks just in time")
            # Bulk create task IDs ahead of time.
            with database.atomic():
                tasks = [Task() for _ in range(len(needs_task_at_index))]
                Task.bulk_create(tasks)

            for i, task in zip(needs_task_at_index, tasks):
                items[i].task_id = task.id
        
        try:
            with database.atomic():
                model.bulk_create(items)
        except IntegrityError:
            log.exception(f"Integrity error when saving results to database.")
            raise a
            for j, item in enumerate(items):
                log.debug(f"Item: {j} {item} {item.__data__}")
        else:
            log.info(f"Saved {len(results)} results to database.")
        
    return None


def _get_return_annotation_output_class(signature):
    """Get the class of the table in the database for this function."""
    ra = signature.return_annotation
    return ra if BaseTaskOutput in getattr(ra, "__mro__", [None]) else ra.__args__[0]


def _is_optional(field):
    return get_origin(field) is Union and type(None) in get_args(field)

def _is_iterable(annotation):
    return getattr(annotation, "_name", None) == "Iterable"

def _is_list_or_tuple(annotation):
    name = getattr(annotation, "_name", None)
    return name in ("List", "Tuple")


def _iterable_parameter_names(signature):
    return [k for k, v in signature.parameters.items() if _is_iterable(v.annotation)]

def _separate_iterable_args(parameters):
    iterable_keys, common_keys = ([], [])
    for k, v in parameters.items():
        if _is_iterable(v.annotation):
            iterable_keys.append(k)
        else:
            common_keys.append(k)
    return (common_keys, iterable_keys)

def _create_parameter_generator(common_args, iterable_keys, iterable_values, fill_value=Ellipsis):
    for v in zip_longest(*iterable_values, fillvalue=fill_value):
        assert fill_value not in v
        yield {**common_args, **dict(zip(iterable_keys, v)) }
    

def _has_many_iterators(function):
    signature = inspect.signature(function)
    _RESTRICTED_KEYS = (_SOURCE_KEY, _DATA_PRODUCT_KEY)

    iterable_parameter_names = _iterable_parameter_names(signature)
    simple = not set(iterable_parameter_names).difference(_RESTRICTED_KEYS)
    return not simple


def _parse_arguments(function, *args, **kwargs):
    signature = inspect.signature(function)
    _RESTRICTED_KEYS = (_SOURCE_KEY, _DATA_PRODUCT_KEY)

    flat_params = dict(zip(signature.parameters, args))

    iterable_parameter_names = _iterable_parameter_names(signature)
    if not set(iterable_parameter_names).difference(_RESTRICTED_KEYS):
        # No iterables. Easy.
        flat_params = dict(zip(signature.parameters, args))
        for key in _RESTRICTED_KEYS:
            flat_params.pop(key, None)
        all_params = repeat(flat_params)
        yield from all_params

    else:
        iterable_kwds = {}
        for k in list(flat_params.keys()):
            if k in iterable_parameter_names:
                iterable_kwds[k] = flat_params.pop(k)
            
        yield from ({ **flat_params, **dict(zip(iterable_kwds, t)) } for t in zip(*iterable_kwds.values()))



def _get_field(parameter):
    annotation = parameter.annotation


    optional = _is_optional(parameter.annotation) 
    if optional:
        check_field_type = get_args(annotation)[0]
    else:
        check_field_type = annotation

    iterable = _is_iterable(check_field_type)
    is_list_or_tuple = _is_list_or_tuple(check_field_type)
    if iterable or is_list_or_tuple:
        check_field_type = get_args(check_field_type)[0]
    
    field_type = {
        int: IntegerField,
        bool: BooleanField,
        str: TextField,
        float: FloatField,  
        dict: JSONField,
        # TODO: shouldn't need to separate this
        Dict: JSONField,
    }[check_field_type]

    try:
        parameter.default
    except:
        has_default = False
    else:
        has_default = True

    kwds = {}
    if optional or (has_default and parameter.default is None):
        kwds["null"] = True
    try:
        kwds["default"] = parameter.default
    except:
        None
    
    if is_list_or_tuple:
        field = ArrayField(field_type, **kwds)
    else:
        field = field_type(**kwds)

    return field

def _get_or_create_database_table_for_task(function):
    signature = inspect.signature(function)
    model = _get_return_annotation_output_class(signature)
    if not model.table_exists():
        # We will add fields to this model based on the function annotations
        for name, parameter in signature.parameters.items():
            if parameter.annotation == inspect._empty:
                print(f"Warning: {name} in {function} has no annotation. Skipping.")
                continue
                
            if name in model._meta.fields or name.lower() in ("data_products", "data_product"):
                print(f"skipping over {name} in {function} table creation")
                continue

            field = _get_field(parameter)

            # Add the field.
            model._meta.add_field(name, field)

        # Create table
        model.create_table()
    
    return model
