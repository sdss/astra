from inspect import isgeneratorfunction
from decorator import decorator
from peewee import chunked, IntegrityError, SqliteDatabase
from playhouse.sqlite_ext import SqliteExtDatabase
from sdsstools.configuration import get_config

from astra.utils import log, Timer

NAME = "astra"
__version__ = "0.6.0"

@decorator
def task(function, *args, **kwargs):
    """
    A decorator for functions that serve as Astra tasks.

    :param function:
        The callable to decorate.

    :param \*args:
        The arguments to the task.

    :param \**kwargs: 
        Keyword arguments for the task and the task decorator. See below.

    :Keyword Arguments:
        * *frequency* (``int``) --
          The number of seconds to wait before saving the results to the database (default: 300).        
        * *result_frequency* (``int``) --
          The number of results  to wait before saving the results to the database (default: 300).        
        * *batch_size* (``int``) --
          The number of rows to insert per batch (default: 1000).
        * *re_raise_exceptions* (``bool``) -- 
          If `True` (default), exceptions raised in the task will be raised. Otherwise, they will be logged and ignored.
    """

    if not isgeneratorfunction(function):
        log.warning(f"Tasks should be generators that `yield` results, but {function} does not `yield`.")

    write_to_database = kwargs.pop("write_to_database", True)
    frequency = kwargs.pop("frequency", 300)
    result_frequency = kwargs.pop("result_frequency", 100_000)
    batch_size = kwargs.pop("batch_size", 1000)
    re_raise_exceptions = kwargs.pop("re_raise_exceptions", True)

    n_results, n_results_since_last_check_point, results = (0, 0, [])
    with Timer(
        function(*args, **kwargs), 
        frequency=frequency, 
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

                try:
                    pk = getattr(result, result._meta.primary_key.name, None)
                except:
                    None
                else:
                    if pk is not None:
                        # already saved from downstream task wrapper
                        # TODO: should we save this?
                        #result.save()
                        yield result                        
                    else:
                        results.append(result)
                        n_results += 1
                        n_results_since_last_check_point += 1
            
            except StopIteration:
                break

            except:
                log.exception(f"Exception raised in task {function.__name__}")        
                if re_raise_exceptions:
                    raise
            
            else:
                if write_to_database and (timer.check_point or n_results_since_last_check_point >= result_frequency):
                    with timer.pause():

                        # Add estimated overheads to each result.
                        timer.add_overheads(results)
                        try:
                            _bulk_insert(results, batch_size, re_raise_exceptions)
                        except:
                            log.exception(f"Exception trying to insert results to database:")
                            if re_raise_exceptions:
                                raise 

                        # We yield here (instead of earlier) because in SQLite the result won't have a
                        # returning ID if we yield earlier. It's fine in PostgreSQL, but we want to 
                        # have consistent behaviour across backends.
                        yield from results
                        log.debug(f"Yielded {len(results)} results")
                        results = [] # avoid memory leak, which can happen if we are running
                        n_results_since_last_check_point = 0

    # It is only at this point that we know:
    # - how many results were created
    # - what the total time elapsed was
    # - what the true cost of overhead time was (before and after yielding results)
    timer.add_overheads(results)
    if write_to_database:
        try:
            # Write any remaining results to the database.
            _bulk_insert(results, batch_size, re_raise_exceptions)
        except:
            log.exception(f"Exception trying to insert results to database:")
            if re_raise_exceptions:
                raise

    yield from results



def _bulk_insert(results, batch_size, re_raise_exceptions=False):
    """
    Insert a batch of results to the database.
    
    :param results:
        A list of records to create (e.g., sub-classes of `astra.models.BaseModel`).
    
    :param batch_size:
        The batch size to use when creating results.
    """
    if not results:
        return None

    log.info(f"Bulk inserting {len(results)} into the database with batch size {batch_size}")

    from astra.models.base import database

    model = results[0].__class__
    if not model.table_exists():
        log.info(f"Creating table {model}")
        model.create_table()
        
    try:
        if isinstance(database, (SqliteExtDatabase, SqliteDatabase)):
            # Do inserts in batches, but make sure that we get the RETURNING id behaviour so that there
            # is consistency in expectations between postgresql/sqlite
            for i, _result in enumerate(database.batch_commit(results, batch_size)):
                # TODO: Not sure why we have to do this,.. but sometimes things try to get re-created?
                if _result.is_dirty():
                    results[i] = model.create(**_result.__data__)
        else:
            try:
                with database.atomic():
                    model.bulk_create(results, batch_size)
            except:
                raise

                    
    except IntegrityError:
        log.exception(f"Integrity error when saving results to database.")
        # Save the entries to a pickle file so we can figure out what went wrong.
        if re_raise_exceptions:
            raise
        else:
            log.warning(f"We will yield the results, but they are not saved.")

    except:
        log.exception(f"Exception occurred when saving results to database.")
        if re_raise_exceptions:
            raise
        else:
            log.warning(f"We will yield the results, but they are not saved.")
    else:
        log.info(f"Saved {len(results)} results to database.")
    
    return None

try:
    config = get_config(NAME)
    
except FileNotFoundError:
    log.exception(f"No configuration file found for {NAME}:")


'''
class Foobar:

    def __init__(self, i):
        self.i = i

@task
def long_task(spectra):
    log.info("in long_task")
    from time import sleep
    sleep(10)
    yield ... 
    for i in range(len(spectra)):
        log.info(f"starting {i}")
        sleep(i)
        yield Foobar(i=i)
    
    yield ... 
    sleep(5)
'''
