import hashlib
import itertools # dat feel
import json
import luigi
import os
import traceback
from time import time

from luigi.mock import MockTarget
from luigi.task_register import Register
from luigi.parameter import ParameterVisibility, _DictParamEncoder, FrozenOrderedDict

from astra.tasks.targets import DatabaseTarget
from astra.utils import log

from packaging.version import parse as parse_version
from astra import __version__


astra_version = parse_version(__version__)


class BaseTask(luigi.Task, metaclass=Register):

    connection_string = luigi.Parameter(
        default="sqlite://", # By default create database in memory.
        config_path=dict(section="task_history", name="db_connection"),
        visibility=luigi.parameter.ParameterVisibility.HIDDEN,
        significant=False
    )

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

        self._warn_on_wrong_param_types(strict)

        str_params = self.to_str_params(only_significant=True, only_public=True)
        self.task_id = task_id_str(
            self.get_task_family(), 
            str_params
        )
        self.__hash = hash(self.task_id)

        self.set_tracking_url = None
        self.set_status_message = None
        self.set_progress_percentage = None
        self._max_params_in_repr = 10


    def _warn_on_wrong_param_types(self, strict=False):
        params = dict(self.get_params())
        batch_param_names = self.batch_param_names()
        for param_name, param_value in self.param_kwargs.items():
            if param_name in batch_param_names and self.is_batch_mode and not strict:
                # Don't warn.
                continue
            params[param_name]._warn_on_wrong_param_type(param_name, param_value)
        

    def __repr__(self):
        """
        Build a task representation like `MyTask(param1=1.5, param2='5')`
        """
        params = self.get_params()
        param_values = self.get_param_values(params, [], self.param_kwargs)

        # Build up task id
        repr_parts = []
        param_objs = dict(params)
        for i, (param_name, param_value) in enumerate(param_values):
            if param_objs[param_name].significant and param_objs[param_name].visibility == ParameterVisibility.PUBLIC:
                if param_name in self.batch_param_names() and self.is_batch_mode:
                    # Show only first two.
                    v = tuple(list(param_value[:2]) + ["..."])
                    v = param_objs[param_name].serialize(v)
                    repr_parts.append(f"{param_name}={v}")
                else:
                    repr_parts.append('%s=%s' % (param_name, param_objs[param_name].serialize(param_value)))
            
            if len(repr_parts) >= self._max_params_in_repr:
                repr_parts.append(f"... et al.")
                break

        task_str = '{}({})'.format(self.get_task_family(), ', '.join(repr_parts))

        return task_str


    def to_str_params(self, only_significant=False, only_public=False):
        """
        Convert all parameters to a str->str hash.
        """
        params_str = {}
        params = dict(self.get_params())
        batch_param_names = self.batch_param_names()
        is_batch_mode = self.is_batch_mode

        for param_name, param_value in self.param_kwargs.items():
            if (((not only_significant) or params[param_name].significant)
                    and ((not only_public) or params[param_name].visibility == ParameterVisibility.PUBLIC)
                    and params[param_name].visibility != ParameterVisibility.PRIVATE):

                if param_name in batch_param_names and is_batch_mode:
                    # TODO: Revisit this.
                    # The reason for doing this is so that we can serialize and de-serialize task parameters
                    # that are batched with tuples. In general you would think that luigi would manage all
                    # the batching and we don't have to do it ourselves, but not when we have dynamic
                    # dependencies. When we have dynamic dependencies that we have to yield() from run(),
                    # the task executes them one at a time, which can be extremely inefficient if there is
                    # overhead related to individual tasks. In that situation we *have* to supply one task
                    # to yield(), which has batch parameters. Those batch parameters then need to get
                    # serialized, which is where we are.
                    params_str[param_name] = json.dumps(param_value, cls=_DictParamEncoder)
                    
                else:
                    params_str[param_name] = params[param_name].serialize(param_value)

        return params_str


    def get_common_param_kwargs(self, klass, include_significant=True):
        common = self.get_common_param_names(klass, include_significant=include_significant)
        return dict([(k, getattr(self, k)) for k in common])


    def get_common_param_names(self, klass, include_significant=True):
        a = self.get_param_names(include_significant=include_significant)
        b = klass.get_param_names(include_significant=include_significant)
        return set(a).intersection(b)
    

    @classmethod
    def from_str_params(cls, params_str):
        """
        Creates an instance from a str->str hash.
        :param params_str: dict of param name -> value as string.
        """
        kwargs = {}
        batch_params = cls.batch_param_names()
        
        for param_name, param in cls.get_params():
            if param_name in params_str:

                param_str = params_str[param_name]
                '''
                if param_name in batch_params:    
                    try:
                        # See notes above about to_str_params().
                        kwargs[param_name] = json.loads(param_str, object_pairs_hook=FrozenOrderedDict)
                    except:

                        print(param_name, param_str, type(param_str))
                        import pickle
                        with open("wtf.pkl", "wb") as fp:
                            pickle.dump((batch_params, params_str, cls.get_params()), fp)
                        
                        raise 
                    else:
                        continue
                '''
                if isinstance(param_str, list):
                    kwargs[param_name] = param._parse_list(param_str)
                else:
                    kwargs[param_name] = param.parse(param_str)
                    
                print(f"PARAM: {param_name} PARSED {kwargs[param_name]} FROM {type(param_str)} and {param_str}")

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
                task = self.__class__(**kwds)

                # Trigger the 'task started' event.
                task.trigger_event(luigi.Event.START, task)
                yield task
        else:
            yield self
    

    def get_batch_size(self):
        """ Get the number of batched tasks. """
        return len(getattr(self, self.batch_param_names()[0])) if self.is_batch_mode else 1


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
    

    def output(self):
        """
        The outputs that the task generates. If all outputs exist then the task is considered done,
        and will not be re-executed.

        For this reason we default to returning a MockTarget based on the task ID and the current
        time. By doing this the task will be re-run every time it is executed, because there is a
        MockTarget (file) that has not been created.

        This should be over-written by sub-classes.
        """
        return MockTarget(f"{self.task_id}_{time()}")


    #def on_success(self):
    #    self.database.touch()

    def on_failure(self, exception):
        #self.database.write_params(self.to_str_params(only_significant=True, only_public=True))

        traceback_string = traceback.format_exc()
        print(f"Traceback caught: {traceback_string}")
        raise
        
        
    def log(self):
        """
        Log something to the database.
        """

        raise NotImplementedError
        

class SDSSDataProduct(luigi.LocalTarget):
    pass


'''
# Event handling.
@luigi.Task.event_handler(luigi.Event.START)
def task_started(task):
    # Write the parameters of this task to the database.
    task.database.write_params(
        task.to_str_params(
            only_significant=True,
            only_public=True
        )
    )

@luigi.Task.event_handler(luigi.Event.SUCCESS)
def task_succeeded(task):
    # We may not want to do this here because it will prevent tasks from being able to re-run. 
    #task.database.touch()
    print(f"Task successful: {task}")
    return None


@luigi.Task.event_handler(luigi.Event.FAILURE)
def task_failed(task, exception):
    print(f"Task failed: {task}")
    print(task)
    print(exception)


@luigi.Task.event_handler(luigi.Event.BROKEN_TASK)
def task_dependencies_failed(task, exception):
    print(f"Task failed on dependencies: {task}")
    print(task)
    print(exception)
'''


def task_id_str(task_family, params, task_id_truncate_hash=10):
    """
    Returns a canonical string used to identify a particular task
    :param task_family: The task family (class name) of the task
    :param params: a dict mapping parameter names to their serialized values
    :return: A unique, shortened identifier corresponding to the family and params
    """
    param_str = json.dumps(params, separators=(',', ':'), sort_keys=True)
    param_hash = hashlib.md5(param_str.encode('utf-8')).hexdigest()
    return f"{task_family}_{param_hash[:task_id_truncate_hash]}"
    
