import hashlib
import itertools # dat feel
import json
import luigi
import os
import traceback
from time import time

from luigi.mock import MockTarget
from luigi.task_register import Register
from luigi.parameter import ParameterVisibility

from astra.tasks.targets import DatabaseTarget
from astra.utils import log



class BaseTask(luigi.Task, metaclass=Register):

    connection_string = luigi.Parameter(
        default="sqlite://", # By default create database in memory.
        config_path=dict(section="task_history", name="db_connection"),
        visibility=luigi.parameter.ParameterVisibility.HIDDEN,
        significant=False
    )

    force = luigi.BoolParameter(significant=False, default=False)
    force_upstream = luigi.BoolParameter(significant=False, default=False)

    def __init__(self, *args, **kwargs):
        super(BaseTask, self).__init__(*args, **kwargs)

        if self.force_upstream is True:
            self.force = True

        if self.force is True:
            done = False
            tasks = [self]
            while not done:
                outputs = luigi.task.flatten(tasks[0].output())
                for out in outputs:
                    log.info(f"Checking {out}")
                    if out.exists():
                        log.warn(f"Removing path {out.path}")
                        print(f"Removing path {out.path}")
                        #os.remove(out.path)
                if self.force_upstream is True:
                    tasks += luigi.task.flatten(tasks[0].requires())
                tasks.pop(0)
                if len(tasks) == 0:
                    done = True
        
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
        for param_name, param_value in self.param_kwargs.items():
            if (((not only_significant) or params[param_name].significant)
                    and ((not only_public) or params[param_name].visibility == ParameterVisibility.PUBLIC)
                    and params[param_name].visibility != ParameterVisibility.PRIVATE):
                params_str[param_name] = params[param_name].serialize(param_value)

        return params_str


    def get_common_param_kwargs(self, klass, include_significant=True):
        common = self.get_common_param_names(klass, include_significant=include_significant)
        return dict([(k, getattr(self, k)) for k in common])


    def get_common_param_names(self, klass, include_significant=True):
        a = self.get_param_names(include_significant=include_significant)
        b = klass.get_param_names(include_significant=include_significant)
        return set(a).intersection(b)
    

    @property
    def is_batch_mode(self):
        try:
            return self._is_batch_mode
        except AttributeError:
            self._is_batch_mode = len(self.batch_param_names()) > 0 \
                and all(
                    isinstance(getattr(self, param_name), tuple) or getattr(self, param_name) is None \
                    for param_name in self.batch_param_names()
                )
        return self._is_batch_mode


    def get_batch_task_kwds(self, include_non_batch_keywords=True):
        kwds = self.param_kwargs.copy() if include_non_batch_keywords else {}
        if not self.batchable:
            yield kwds

        else:
            batch_param_names = self.batch_param_names()

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
    
