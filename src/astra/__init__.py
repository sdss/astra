from inspect import isgeneratorfunction
from decorator import decorator
from peewee import IntegrityError
from sdsstools.configuration import get_config

from astra.utils import log, Timer

NAME = "astra"
__version__ = "0.7.0"

@decorator
def task(
    function, 
    *args, 
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

    :param \*args:
        The arguments to the task.

    :param batch_size: [optional]
        The number of rows to insert per batch (default: 1000).
    
    :param write_frequency: [optional]
        The number of seconds to wait before saving the results to the database (default: 300).
    
    :param write_to_database: [optional]
        If `True` (default), results will be written to the database. Otherwise, they will be ignored.

    :param re_raise_exceptions: [optional]
        If `True` (default), exceptions raised in the task will be raised. Otherwise, they will be logged and ignored.

    :param \**kwargs: 
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
                # `Ellipsis` has a special meaning to Astra tasks.
                # It is a marker that tells the Astra timer that the interval spent so far is related
                # to common overheads, not specifically to the calculations of one result.
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
    log.info(f"Bulk inserting {len(results)} into the database")

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
    
try:
    config = get_config(NAME)
    
except FileNotFoundError:
    log.exception(f"No configuration file found for {NAME}:")