from inspect import isgeneratorfunction
from decorator import decorator
from peewee import IntegrityError, JOIN
from sdsstools.configuration import get_config

from astra.utils import log, Timer, resolve_task, resolve_model, get_return_type, expects_spectrum_types, version_string_to_integer, get_task_group_by_string

NAME = "astra"
__version__ = "0.7.0"

@decorator
def task(
    function, 
    *args, 
    group_by=None,
    batch_size: int = 1000,
    write_frequency: int = 300,
    write_to_database: bool = True,
    re_raise_exceptions: bool = True, 
    **kwargs
):
    """
    A decorator for functions that serve as Astra tasks.

    :param function:
        The callable to decorate.

    :param *args:
        The arguments to the task.

    :param batch_size: [optional]
        The number of rows to insert per batch (default: 1000).
    
    :param write_frequency: [optional]
        The number of seconds to wait before saving the results to the database (default: 300).
    
    :param write_to_database: [optional]
        If `True` (default), results will be written to the database. Otherwise, they will be ignored.

    :param re_raise_exceptions: [optional]
        If `True` (default), exceptions raised in the task will be raised. Otherwise, they will be logged and ignored.

    :param **kwargs: 
        Keyword arguments for the task and the task decorator. See below.
    """

    if not isgeneratorfunction(function):
        log.warning(f"Tasks should be generators that `yield` results, but {function} does not `yield`.")

    n, results = (0, [])
    with Timer(
        function(*args, **kwargs), 
        frequency=write_frequency, 
        attr_t_elapsed="t_elapsed",
        attr_t_overhead="t_overhead",
    ) as timer:
        while True:
            try:
                result = next(timer)

                if result is Ellipsis:
                    continue

                if not write_to_database:
                    yield result
                else:
                    results.append(result)
                    n += 1
            
            except StopIteration:
                break

            except:
                log.exception(f"Exception raised in task {function.__name__}")        
                if re_raise_exceptions:
                    raise

            finally:
                if write_to_database and (timer.check_point or n >= batch_size):
                    with timer.pause():
                        # Add estimated overheads to each result.
                        timer.add_overheads(results)
                        try:
                            yield from bulk_insert_or_replace_pipeline_results(results)
                        except:
                            log.exception(f"Exception trying to insert results to database:")
                            if re_raise_exceptions:
                                raise 
                        finally:
                            results = [] # avoid memory leak, which can happen if we are running
                            n = 0

    # It is only at this point that we know:
    # - how many results were created
    # - what the total time elapsed was
    # - what the true cost of overhead time was (before and after yielding results)
    if n > 0:
        if write_to_database:
            timer.add_overheads(results)
            try:
                # Write any remaining results to the database.
                yield from bulk_insert_or_replace_pipeline_results(results)
            except:
                log.exception(f"Exception trying to insert results to database:")
                if re_raise_exceptions:
                    raise


def bulk_insert_or_replace_pipeline_results(results):
    """
    Insert a batch of results to the database.
    
    :param results:
        A list of records to create (e.g., sub-classes of `astra.models.BaseModel`).    
    """
    #log.info(f"Bulk inserting {len(results)} into the database")

    if len(results) == 0:
        return 
        
    first = results[0]
    database, model = (first._meta.database, first.__class__)
    # Here, `preserve` is the set of fields that we want to overwrite if there is a conflict.
    # We do not want to overwrite `task_pk` or `created`, and `v_astra_major_minor` is a generated field.
    preserve = list(set(model._meta.fields.values()) - {model.task_pk, model.created, model.v_astra_major_minor})

    # We cannot guarantee that the order of the RETURNING clause will be the same as the input order.
    # We need the `task_pk` generated, and we need (`spectrum_pk`, `v_astra`) to map to the results list, and we need
    # `created` so that we can attach it to the object.        
    q = (
        model
        .insert_many(r.__data__ for r in results)
        .returning(
            model.task_pk, 
            model.spectrum_pk, 
            model.v_astra, 
            model.created
        )
        .on_conflict(
            conflict_target=[model.spectrum_pk, model.v_astra_major_minor],
            preserve=preserve
        )
        .tuples()
        .execute()
    )
    results_dict = {(r.spectrum_pk, r.v_astra // 1000): r for r in results}
    for task_pk, spectrum_pk, v_astra, created in q:
        r = results_dict.pop((spectrum_pk, v_astra // 1000))
        r.__data__.update(task_pk=task_pk, created=created)
        r._dirty.clear()
        yield r

    if len(results_dict) > 0:
        raise IntegrityError("Failed to insert all results into the database.")




def generate_queries_for_task(task, input_model=None, limit=None, page=None):
    """
    Generate queries for input data that need to be processed by the given task.

    :param task:
        The task name, or callable.
    
    :param input_model: [optional]
        The input spectrum model. If `None` is given then a query will be generated for each
        spectrum model expected by the task, based on the task function signature.
    
    :param limit: [optional]
        Limit the number of rows for each spectrum model query.
    """
    from astra.models.source import Source
    from astra.models.spectrum import Spectrum

    current_version = version_string_to_integer(__version__) // 1000

    fun = resolve_task(task)

    input_models = expects_spectrum_types(fun) if input_model is None else (resolve_model(input_model), )
    output_model = get_return_type(fun)
    group_by_string = get_task_group_by_string(fun)
    
    for input_model in input_models:
        where = (
            output_model.spectrum_pk.is_null()
        |   (input_model.modified > output_model.modified)
        )
        on = (
            (output_model.v_astra_major_minor == current_version)
        &   (input_model.spectrum_pk == output_model.spectrum_pk)
        )
        q = (
            input_model
            .select(input_model, Source)
            .join(Source, attr="source")
            .switch(input_model)
            .join(Spectrum)
            .join(output_model, JOIN.LEFT_OUTER, on=on)
            .where(where)
        )
        if group_by_string is not None:
            group_by_resolved = []
            for item in eval(group_by_string):
                if isinstance(item, str):
                    group_by_resolved.append(getattr(input_model, item))
                else:
                    group_by_resolved.append(item)
            
            q = q.group_by(*group_by_resolved)

        if limit is not None:
            if page is not None:
                q = q.paginate(page, limit)
            else:
                q = q.limit(limit)
        yield (input_model, q)


try:
    config = get_config(NAME)
    
except FileNotFoundError:
    log.exception(f"No configuration file found for {NAME}:")