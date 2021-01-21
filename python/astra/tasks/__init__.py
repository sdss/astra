import hashlib
import itertools # dat feel
import json
import luigi
import os
import traceback
from datetime import datetime

from luigi.task_register import Register
from luigi.parameter import ParameterVisibility, _DictParamEncoder, FrozenOrderedDict
from packaging.version import parse as parse_version
from astra.utils import log
from astra import __version__
from tqdm import tqdm

from astra.database import database
from astra.database.astradb import TaskState, TaskParameter

session = database.Session()

astra_version = parse_version(__version__)


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
        
        strict = kwargs.pop("strict", False)

        params = self.get_params()
        param_values = self.get_param_values(params, args, kwargs)

        # Set all values on class instance
        for key, value in param_values:
            setattr(self, key, value)

        # Register kwargs as an attribute on the class. Might be useful
        self.param_kwargs = dict(param_values)

        self._database_parameter_kwargs = {}
        # Here are some parameters that we know will exist.
        # We only record parameters that are significant and not batchable.
        #ignore = ("release", "astra_version_major", "astra_version_minor")
        ignore = ()
        for k, p in params:
            if p.significant and not p._is_batchable() and k not in ignore:
                self._database_parameter_kwargs[k] = self.param_kwargs[k]

        self._database_parameter_pk = self.get_parameter_pk()
        
        self._warn_on_wrong_param_types(strict)

        str_params = self.to_str_params(only_significant=True, only_public=False)
        self.task_id = task_id_str(
            self.get_task_family(), 
            str_params
        )
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

        return f"<{task_family}({task_hash}, batch_size={batch_size})>"


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
                params_str[k] = json.dumps(v, cls=_DictParamEncoder)
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
                            value = param.parse(param_str)
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
        return []


    def output(self):
        return []


    def complete(self):
        if self.strict_output_checking:
            return super(BaseTask, self).complete()

        else:
            instance = self.get_state_instance()
            return False if instance is None else (instance.status_code == 1)
    


    def get_state_query(self, full_output=False):
        kwds = dict(
            task_module=self.task_module,
            task_id=self.task_id
        )
        model = TaskState
        q = session.query(model).filter_by(**kwds)
        if full_output:
            return (q, model, kwds)
        return q



    def get_state_instance(self, full_output=False):
        q, model, kwds = self.get_state_query(full_output=True)
        instance = q.one_or_none()
        if full_output:
            return (instance, model, kwds)
        return instance
        

    def get_or_create_state_instance(self, defaults=None):
        instance, model, kwds = self.get_state_instance(full_output=True)
        create = (instance is None)
        if create:
            if defaults is not None:
                kwds.update(defaults)

            instance = model(**kwds)
            with session.begin():
                session.add(instance)
            
        return (instance, create)


    def delete_state(self):
        return self.get_state_query().delete()

     
    def update_state(self, **kwargs):
        return self.get_state_query().update(kwargs)
        

    def get_parameter_pk(self):
        parameters = self._database_parameter_kwargs
        parameter_hash = hashify(parameters)
        return int(parameter_hash, 16)


    def get_or_create_parameter_instance_pk(self):
        """
        Return the primary key of the task parameter row in the database for this task.
        In the database we only store task parameters that are significant and not
        batchable.
        """

        pk = self.get_parameter_pk()
        parameters = self._database_parameter_kwargs
        model = TaskParameter

        q = session.query(model.pk).filter_by(pk=pk)
        if q.one_or_none() is None:
            instance = model(pk=pk, parameters=parameters)
            with session.begin():
                session.add(instance)

        return pk


    def trigger_event_start(self):
        return self.trigger_event(luigi.Event.START, self)


    def trigger_event_succeeded(self):
        return self.trigger_event(luigi.Event.SUCCESS, self)
    

    def trigger_event_processing_time(self, duration, cascade=False):
        """
        Trigger the event that signals the processing time of the event.

        :param duration:
            The time taken for this event.
        
        :param cascade: [optional]
            Also trigger the task succeeded event (default: False).
        """
        self.trigger_event(luigi.Event.PROCESSING_TIME, self, duration)
        if cascade:
            self.trigger_event_succeeded()




    



class SDSSDataProduct(luigi.LocalTarget):
    pass


# TODO: Better alignment between event names and return codes needed.

@BaseTask.event_handler(luigi.Event.START)
def task_started(task):

    def trigger(t, parameter_pk):
            
        instance, model, state_kwds = t.get_state_instance(full_output=True)

        kwds = dict(
            status_code=0,
            created=datetime.now(),
            modified=datetime.now(),
            completed=None,
            parameter_pk=parameter_pk
        )

        if instance is None:
            if not t.is_batch_mode:
                # Get any relations to data model products (apogee_star, boss_spec, etc).
                try:
                    data_model_kwds = t.get_or_create_data_model_relationships()
                except AttributeError:
                    None
                else:
                    kwds.update(data_model_kwds)

            # Create instance.
            kwds.update(state_kwds)
            instance = model(**kwds)
            with session.begin():
                session.add(instance)

        else:
            # If this task is completed, then we should just update the modified time.
            if instance.status_code == 1:
                t.update_state(modified=datetime.now())
            
            else:
                # Update instance.
                t.update_state(**kwds)

        return instance

    # Store parameters.
    parameter_pk = task.get_or_create_parameter_instance_pk()

    trigger(task, parameter_pk)

    # Create sub-tasks with the parameter pk from the primary.
    if task.is_batch_mode:
        for each in tqdm(task.get_batch_tasks(), total=task.get_batch_size(), desc="Trigger tasks (start)"):
            trigger(each, parameter_pk)
    

@BaseTask.event_handler(luigi.Event.SUCCESS)
def task_succeeded(task):
    """
    Mark a task as being complete in the database.

    :param task:
        The completed task.
    """
    state_factory = lambda **_: dict(
        status_code=1,
        completed=datetime.now(),
        modified=datetime.now()
    )
    if task.is_batch_mode:
        for each in tqdm(task.get_batch_tasks(), total=task.get_batch_size(), desc="Trigger tasks (complete)"):
            each.update_state(**state_factory())
            
    task.update_state(**state_factory())
    

@BaseTask.event_handler(luigi.Event.FAILURE)
def task_failed(task, args):
    # TODO: trigger on sub-tasks too?
    task.get_or_create_state_instance()
    task.update_state(
        status_code=30,
        modified=datetime.now()
    )


@BaseTask.event_handler(luigi.Event.PROCESS_FAILURE)
def task_process_failed(task):
    # TODO: trigger on sub-tasks too?
    task.get_or_create_state_instance()
    task.update_state(
        status_code=40,
        modified=datetime.now()
    )


@BaseTask.event_handler(luigi.Event.BROKEN_TASK)
def task_broken(task):
    # TODO: trigger on sub-tasks too?
    task.get_or_create_state_instance()
    task.update_state(
        status_code=40,
        modified=datetime.now()
    )

@BaseTask.event_handler(luigi.Event.DEPENDENCY_MISSING)
# TODO: trigger on sub-tasks too?
def task_dependency_missing(task):
    task.get_or_create_state_instance()
    task.update_state(
        status_code=40,
        modified=datetime.now()
    )

@BaseTask.event_handler(luigi.Event.PROCESSING_TIME)
def task_processing_time(task, duration):
    # TODO: trigger on sub-tasks too?
    task.update_state(
        duration=duration,
        modified=datetime.now()
    )


def hashify(params, max_length=8):
    param_str = json.dumps(params, cls=_DictParamEncoder, separators=(',', ':'), sort_keys=True)
    param_hash = hashlib.md5(param_str.encode('utf-8')).hexdigest()
    return param_hash[:max_length]

    



def task_id_str(task_family, params):
    """
    Returns a canonical string used to identify a particular task
    :param task_family: The task family (class name) of the task
    :param params: a dict mapping parameter names to their serialized values
    :return: A unique, shortened identifier corresponding to the family and params
    """
    param_hash = hashify(params)
    return f"{task_family}_{param_hash}"
    
