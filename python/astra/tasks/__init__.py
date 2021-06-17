import hashlib
import itertools # dat feel
import json
import luigi
import os
import threading

import traceback
from datetime import datetime

from luigi.task import flatten
from luigi.task_register import Register
from luigi.parameter import ParameterVisibility, _DictParamEncoder, FrozenOrderedDict
from packaging.version import parse as parse_version
from astra.utils import log
from astra import __version__
from tqdm import tqdm
from sdsstools.logger import get_exception_formatted

from astra.database import session, astradb
from astra.tasks.utils import (hashify, task_id_str)

astra_version = parse_version(__version__)

last_task_exception = None

class BaseTask(luigi.Task, metaclass=Register):

    """ A base task class for Astra. """

    # Astra versioning. These parameters should never be changed directly by the user.
    # We assume that major and minor version changes should cause all tasks to re-run
    # (e.g., a major version for a data release, minor version change for serious bug fixes,
    # and micro or dev version changes for bug fixes that do not affect all tasks.)
    astra_version_major = luigi.IntParameter(
        default=astra_version.major, 
        significant=True,
        visibility=luigi.parameter.ParameterVisibility.HIDDEN
    )
    astra_version_minor = luigi.IntParameter(
        default=astra_version.minor,
        significant=True,
        visibility=luigi.parameter.ParameterVisibility.HIDDEN
    )
    astra_version_micro = luigi.IntParameter(
        default=astra_version.micro, 
        significant=False,
        visibility=luigi.parameter.ParameterVisibility.HIDDEN
    )
    astra_version_dev = luigi.BoolParameter(
        default=astra_version.dev is not None, 
        significant=False,
        visibility=luigi.parameter.ParameterVisibility.HIDDEN
    )

    strict_output_checking = luigi.BoolParameter(
        default=False,
        significant=False,
        visibility=luigi.parameter.ParameterVisibility.HIDDEN
    )

    def __init__(self, *args, **kwargs):
        super(BaseTask, self).__init__(*args, **kwargs)
        

        params = self.get_params()
        param_values = self.get_param_values(params, args, kwargs)

        # Set all values on class instance
        for key, value in param_values:
            setattr(self, key, value)

        # Register kwargs as an attribute on the class. Might be useful
        self.param_kwargs = dict(param_values)
        
        self._warn_on_wrong_param_types()

        str_params = self.to_str_params(only_significant=True, only_public=False)
        self.task_id = task_id_str(
            self.get_task_family(), 
            str_params
        )

        self.__str_params = str_params
        self.__hash = hash(self.task_id)

        self.set_tracking_url = None
        self.set_status_message = None
        self.set_progress_percentage = None


    def _warn_on_wrong_param_types(self, strict=False):
        params = dict(self.get_params())
        batch_param_names = self.batch_param_names()
        for param_name, param_value in self.param_kwargs.items():
            if param_name in batch_param_names and self.is_batch_mode and not strict:
                # Don't warn.
                continue
            params[param_name]._warn_on_wrong_param_type(param_name, param_value)
        

    def __repr__(self):
        """ Build a task representation like `MyTask(hash: param1=1.5, param2='5')` """

        task_family, task_hash = self.task_id.split("_")
        batch_size = self.get_batch_size()

        if batch_size > 1:
            return f"<{task_family}({task_hash}, batch_size={batch_size})>"
        else:
            return f"<{task_family}({task_hash})>"


    def get_common_param_kwargs(self, klass, include_significant=True):
        common = self.get_common_param_names(klass, include_significant=include_significant)
        return dict([(k, getattr(self, k)) for k in common])


    def get_common_param_names(self, klass, include_significant=True):
        a = self.get_param_names(include_significant=include_significant)
        b = klass.get_param_names(include_significant=include_significant)
        return set(a).intersection(b)


    def get_hashed_params(self, only_significant=True, only_public=False):

        hashed_params = {}
        params = dict(self.get_params())
        for param_name, param_value in self.param_kwargs.items():
            if (((not only_significant) or params[param_name].significant)
                    and ((not only_public) or params[param_name].visibility == ParameterVisibility.PUBLIC)
                    and params[param_name].visibility != ParameterVisibility.PRIVATE):

                hashed_params[param_name] = param_value

        return hashed_params
    

    def to_str_params(self, only_significant=True, only_public=False):
        """
        Convert all parameters to a str->str hash.
        """

        hashed_params = self.get_hashed_params(only_significant=only_significant, only_public=only_public)

        params_str = {}
        params = dict(self.get_params())
        batch_param_names = self.batch_param_names()
        is_batch_mode = self.is_batch_mode

        for k, v in hashed_params.items():
            if k in batch_param_names and is_batch_mode:
                params_str[k] = json.dumps(v, cls=_DictParamEncoder, separators=(',', ':'), sort_keys=True)
            else:
                params_str[k] = params[k].serialize(v)
                
        return params_str
    

    @classmethod
    def from_str_params(cls, params_str):
        """
        Creates an instance from a str->str hash.
        :param params_str: dict of param name -> value as string.
        """
        kwargs = {}
        batch_param_names = cls.batch_param_names()
        
        for param_name, param in cls.get_params():
            if param_name in params_str:

                # JSON will dump tuples and lists as if they are lists,
                # and will load lists as lists even if they were tuples.

                param_str = params_str[param_name]

                # A "string" parameter will be parsed by param.parse as if it is a single
                # string value, even if it is actually a JSON dump of a tuple with many strings.
                if param_name in batch_param_names:
                    # Duck-type test for whether this is batch mode.
                    try:
                        value = json.loads(param_str, object_pairs_hook=FrozenOrderedDict)
                    except:
                        if isinstance(param_str, list) and not isinstance(param, luigi.ListParameter):
                            value = tuple(list(map(param.parse, param_str)))
                        
                        else:
                            try:
                                value = param.parse(param_str)
                            except:
                                # We also have to remove commas from the right hand side, as
                                # (324.,) is a valid way to represent a tuple, but [324.,]
                                # is not a valid way to load a list through JSON.
                                value = tuple(json.loads(f"[{param_str[1:-1].rstrip(',')}]"))
                    else:
                        if isinstance(value, list) and not isinstance(param, luigi.ListParameter):
                            value = tuple(value)
                        else:
                            value = param.parse(value)
                else:
                    value = param.parse(param_str)
                
                kwargs[param_name] = value

        return cls(**kwargs)


    @property
    def is_batch_mode(self):
        """ A boolean property indicating whether the task is in batch mode or not. """
        try:
            return self._is_batch_mode
        except AttributeError:
            self._is_batch_mode = len(self.batch_param_names()) > 0 \
                and all(
                    isinstance(getattr(self, param_name), (list, tuple)) or getattr(self, param_name) is None \
                    for param_name in self.batch_param_names()
                )
        return self._is_batch_mode


    def get_batch_task_kwds(self, include_non_batch_keywords=True):
        
        batch_param_names = self.batch_param_names()

        if self.is_batch_mode:
            kwds = self.param_kwargs.copy() if include_non_batch_keywords else {}

            all_batch_kwds = {}
            
            for param_name in batch_param_names:
                entry = getattr(self, param_name)
                if isinstance(entry, tuple):
                    all_batch_kwds[param_name] = entry
                else:
                    # Some batch parameters may not be given as a tuple.
                    # Cycle indefinitely over this value. This is equivalent to repeating it N times,
                    # where N would be the number of batch parameters given elsewhere.
                    all_batch_kwds[param_name] = itertools.cycle((entry, ))

            for batch_values in zip(*map(all_batch_kwds.get, batch_param_names)):
                yield { **kwds, **dict(zip(batch_param_names, batch_values)) }

        else:
            # We have no batch keywords yet.
            kwds = { k: v for k, v in self.param_kwargs.items() if k in batch_param_names or include_non_batch_keywords }
            yield kwds


    def get_batch_tasks(self):
        """ A generator that yields task(s) that are to be run. Works in single or batch mode. """
        # This task can be run in single mode or batch mode.
        if self.is_batch_mode:
            for kwds in self.get_batch_task_kwds():
                yield self.__class__(**kwds)
        else:
            yield self
    

    def get_batch_size(self):
        """ Get the number of batched tasks. """
        try:
            return self._batch_size

        except AttributeError:
            if self.is_batch_mode:
                self._batch_size = max(len(getattr(self, pn)) for pn in self.batch_param_names())
            else:
                self._batch_size = 1
            
        finally:
            return self._batch_size


    @property
    def output_base_dir(self):
        """ Base directory for storing task outputs. """
        # TODO: When you have settled on a data model, put it into the Tree and replace this!
        return os.path.join(
            os.environ.get("SAS_BASE_DIR"),
            "sdsswork",
            "mwm",
            "astra",
            f"{self.astra_version_major}.{self.astra_version_minor}.{self.astra_version_micro}",
            self.task_namespace
        )

    
    def get_input(self, key):
        """
        Return a single input from the task, assuming the inputs are a dictionary.
        This can be performed by using `task.input()[key]`, but when there are many inputs
        (e.g., in batch mode), this can be unnecessarily slow.

        :param key:
            The key of the requirements dictionary to return.
        """
        return self.requires()[key].output()
        

    def requires(self):
        """ The requirements of this task. """
        return []


    def output(self):
        """ The outputs of this task. """
        return []


    def query_state(self, full_output=False):
        """ 
        Query the database for this task and return the SQLAlchemy ORM Query.

        :param full_output: [optional]
            Optionally return a three-length tuple containing the ORM query, database model, and 
            keywords to filter by.
        """
        kwds = dict(
            task_module=self.task_module,
            task_id=self.task_id
        )
        model = astradb.Task
        q = session.query(model).filter_by(**kwds)
        if full_output:
            return (q, model, kwds)
        return q


    def get_or_create_state(self, defaults=None):
        """
        Get (or create) an entry in the database for this task. 

        Note that this will only create an entry for the *task*, and not for the parameters of the
        task. This is useful when creating many task entries, with the intent you will create the
        parameter entries later, and you want to minimise overhead. If you want to create an entry
        for this task *and* the parameters, use `create_state()`.

        This function returns a two-length tuple containing the SQLAlchemy instance, and a boolean
        flag indicating whether the entry was created (True) or just retrieved (False).

        :param defaults: [optional]
            A dictionary of default key, value pairs to provide if the entry needs to be created in
            the database.
        """
        
        q, model, kwds = self.query_state(full_output=True)
        instance = q.one_or_none()
        create = (instance is None)
        if create:
            defaults = defaults or {}
            instance = model(**{**defaults, **kwds})
            with session.begin():
                session.add(instance)
        
        session.flush()
        return (instance, create)


    def create_state(self):
        """ Create an entry in the database for this task, and its parameters. """
        
        # Create parameter instances first.
        parameter_pks = {}
        params = dict(self.get_params())
        for key, values in self.param_kwargs.items():
            if params[key].significant:
                as_list = (self.is_batch_mode and params[key]._is_batchable())
                if not as_list:
                    values = (values, )
                
                pks = []
                for value in values:
                    kwds = dict(
                        parameter_name=key,
                        parameter_value=params[key].serialize(value)
                    )
                    parameter_instance, _ = get_or_create_parameter_instance(**kwds)
                    pks.append(parameter_instance.pk)
                
                parameter_pks[key] = pks if as_list else pks[0]
                
                
        # Create task instances.
        task_pks = []
        for i, task in enumerate(self.get_batch_tasks()):
            instance, _ = task.get_or_create_state()
            task_pks.append(instance.pk)

            # Reference parameter keys.
            for key, pks in parameter_pks.items():
                pk = pks if isinstance(pks, int) else pks[i]
                kwds = dict(task_pk=instance.pk, parameter_pk=pk)

                q = session.query(astradb.TaskParameter).filter_by(**kwds)
                if q.one_or_none() is None:
                    with session.begin():
                        session.add(astradb.TaskParameter(**kwds))

        if self.is_batch_mode:
            parent_task, _ = self.get_or_create_state()

            # Reference batch identifiers.
            for child_task_pk in task_pks:
                kwds = dict(
                    parent_task_pk=parent_task.pk,
                    child_task_pk=child_task_pk
                )                    
                q = session.query(astradb.BatchInterface).filter_by(**kwds)
                if q.one_or_none() is None:
                    with session.begin():
                        session.add(astradb.BatchInterface(**kwds))
    
        session.flush()
        return (parameter_pks, task_pks)


    def delete_state(self, cascade=False):
        """
        Delete this task entry in the database.

        :param cascade: [optional]
            Cascade this to any tasks in this batch.
        """
        if not cascade:
            return self.query_state().delete()
        else:
            for task in self.get_batch_tasks():
                task.delete_state()
        return None
             

    def update_state(self, state, cascade=False):
        """
        Update the task entry in the database with the given state dictionary.

        :param cascade: [optional]
            Cascade this to any tasks in this batch.
        """
        if not cascade:
            return self.query_state().update(state)
        else:
            for task in self.get_batch_tasks():
                task.query_state().update(state)
        session.flush()
        return None

                
    def trigger_event_start(self):
        """ Trigger an event signalling that the task has started. """
        return self.trigger_event(luigi.Event.START, self)


    def trigger_event_succeeded(self):
        """ Trigger an event signalling that the task has succeeded. """
        return self.trigger_event(luigi.Event.SUCCESS, self)


    def trigger_event_failed(self):
        """ Trigger an event signalling that the task has failed. """
        return self.trigger_event(luigi.Event.FAILURE, self)


    def trigger_event_processing_time(self, duration, cascade=False):
        """
        Trigger the event that signals the processing time of the event.

        :param duration:
            The time taken for this event.
        
        :param cascade: [optional]
            Also trigger the task succeeded event (default: False).
        """
        triggered = self.trigger_event(luigi.Event.PROCESSING_TIME, self, duration)
        if cascade:
            self.trigger_event_succeeded()
        return triggered


def get_or_create_parameter_instance(parameter_name, parameter_value):
    """
    Get (or create) an instance in the database for this parameter name and value pair.

    Returns a two-length tuple containing the `astradb.Parameter` instance, and a boolean flag
    indicating whether the instance was created (True) or just retrieved (False).

    :param parameter_name:
        The name of the parameter in the task.
    
    :param parameter_value:
        The value given to that parameter name.
    """
    kwds = dict(parameter_name=parameter_name, parameter_value=parameter_value)
    q = session.query(astradb.Parameter).filter_by(**kwds)
    instance = q.one_or_none()
    create = (instance is None)
    if create:
        instance = astradb.Parameter(**kwds)
        with session.begin():
            session.add(instance)
    return (instance, create)


def _create_and_announce_state(task):
    log.debug(f"Creating state for {task}")
    task.create_state()
    log.debug(f"Created state for {task}")


@BaseTask.event_handler(luigi.Event.START)
def task_started(task):
    """
    Create entries in the database for this task and it's parameters. 
    
    :param task:
        The started task.
    """
    thread = threading.Thread(
            target=_create_and_announce_state,
            args=(task, ),
            #daemon=True
    )
    thread.start()
    # See https://stackoverflow.com/questions/49134260/what-happens-to-threads-in-python-when-there-is-no-join
    #thread.join()
    
    

@BaseTask.event_handler(luigi.Event.SUCCESS)
def task_succeeded(task):
    """
    Mark a task as being complete in the database.

    :param task:
        The completed task.
    """
    task.update_state(
        dict(status_code=1, modified=datetime.now()),
        cascade=True
    )
    

@BaseTask.event_handler(luigi.Event.FAILURE)
def task_failed(task, exception):
    # TODO: trigger on sub-tasks too?
    """
    Mark a task as failed in the database.

    :param task:
        The failed task.
    """
    # Store this as the last exception raised by a task,
    # in case we want to dive into the code with:
    #
    #   raise astra.tasks.last_task_exception
 
    global last_task_exception
    last_task_exception = exception

    # Log the exception.
    log.error(get_exception_formatted(
        type(exception),
        exception,
        exception.__traceback__
    ))

    # Update the state in the database.
    #parameter_pks, task_pks = task.create_state()
    task.update_state(
        dict(status_code=30, modified=datetime.now())
    )
    return None


@BaseTask.event_handler(luigi.Event.PROCESS_FAILURE)
def task_process_failed(task):
    """
    Mark a task process as failed in the database.

    :param task:
        The failed task.
    """
    # TODO: trigger on sub-tasks too?
    task.update_state(
        dict(status_code=40, modified=datetime.now())
    )


@BaseTask.event_handler(luigi.Event.BROKEN_TASK)
def task_broken(task):
    """
    Mark a task as being broken in the database.

    :param task:
        The broken task.
    """    
    task.update_state(
        dict(status_code=50, modified=datetime.now())
    )


@BaseTask.event_handler(luigi.Event.DEPENDENCY_MISSING)
def task_dependency_missing(task):
    """
    Mark a task as having missing dependencies in the database.

    :param task:
        The task with missing dependencies.
    """
    # TODO: trigger on sub-tasks too?
    task.update_state(
        dict(status_code=60, modified=datetime.now())
    )


@BaseTask.event_handler(luigi.Event.PROCESSING_TIME)
def task_processing_time(task, duration):
    """
    Record the processing duration of a task in the database.

    :param task:
        The completed task.

    :param duration:
        The duration of the task in seconds.
    """
    # TODO: trigger on sub-tasks too?
    task.update_state(dict(
        duration=duration,
        modified=datetime.now()
    ))