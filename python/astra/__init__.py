from inspect import isgeneratorfunction
from decorator import decorator
from peewee import IntegrityError
from sdsstools.configuration import get_config

from astra.utils import log, Timer

NAME = "astra"
__version__ = "0.4.0"

@decorator
def task(
    function, 
    frequency=300,
    batch_size=1000,
    re_raise_exceptions=True, # TODO: figure out how tomake this logic false for SLURM jobs, or AIRFLOW tasks, but true for interactive sessions
    use_slurm=False,
    slurm_kwargs=None,
    slurm_default_profile=None,
    *args,
    **kwargs
):
    """
    A decorator for functions that serve as Astra tasks.

    :param function:
        The callable to decorate.

    :param frequency: [optional]
        The number of seconds to wait before saving the results to the database.
    
    :param batch_size: [optional]
        The number of rows to insert per batch.
    
    :param re_raise_exceptions: [optional]
        If True, exceptions raised in the task will be raised. Otherwise, they will be logged and ignored.
    """
    
    if not isgeneratorfunction(function):
        raise TypeError("Task functions must be generators.")

    # Check whether we get any slurm things.
    if use_slurm:
        raise NotImplementedError

    f = function(*args, **kwargs)

    results = []
    with Timer(f, frequency) as timer:
        while True:
            try:
                result = next(timer)
                results.append(result)
                
            except StopIteration:
                break

            except:
                log.exception(f"Exception raised in task {function.__name__}")        
                if re_raise_exceptions:
                    raise
            
            else:
                yield result

                if timer.check_point:
                    with timer.pause():
                        _bulk_create_results(results, batch_size)
                        results = [] # avoid memory leak
                        
    # Write any remaining results to the database.
    _bulk_create_results(results, batch_size)

    return None


def _bulk_create_results(results, batch_size):
    if not results:
        return None
    
    from astra.models import database

    model = results[0].__class__
    if not model.table_exists():
        log.info(f"Creating table {model}")
        model.create_table()
    
    try:
        with database.atomic():
            model.bulk_create(results, batch_size=batch_size)

    except IntegrityError:
        log.exception(f"Integrity error when saving results to database.")
        # Save the entries to a pickle file so we can figure out what went wrong.
        raise
    
    else:
        log.info(f"Saved {len(results)} results to database.")
                
    return None

try:
    config = get_config(NAME)
    
except FileNotFoundError:
    log.exception(f"No configuration file found for {NAME}:")
