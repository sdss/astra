import abc
import os
import inspect
import json
import numpy as np
from time import time
from astra import (log, __version__)
from astra.utils import (flatten, )# timer)
from astra.database.astradb import (DataProduct, Task, TaskInputDataProducts, Bundle, TaskBundle)

class Parameter:
    def __init__(self, name, bundled=False, **kwargs):
        self.name = name
        self.bundled = bundled
        if "default" in kwargs:
            self.default = kwargs.get("default")
        


class ExecutableTaskMeta(abc.ABCMeta):
    def __new__(cls, *args):
        new_class = super(ExecutableTaskMeta, cls).__new__(cls, *args)
        new_class.execute = new_class._execute_decorator(new_class.execute)
        return new_class


def get_or_create_data_products(iterable):
    if iterable is None:
        return None

    if isinstance(iterable, str):
        try:
            iterable = json.loads(iterable)
        except:
            pass

    dps = []
    for dp in iterable:
        if isinstance(dp, DataProduct) or dp is None:
            dps.append(dp)
        elif isinstance(dp, str) and os.path.exists(dp):
            dp, _ = DataProduct.get_or_create(
                release=None,
                filetype="full",
                kwargs=dict(full=dp)
            )
            dps.append(dp)
        elif isinstance(dp, (list, tuple, set)):
            dps.append(get_or_create_data_products(dp))
        else:
            try:
                dp = DataProduct.get(pk=int(dp))
            except:
                raise TypeError(f"Unknown input data product type ({type(dp)}): {dp}")
            else:
                dps.append(dp)
    
    return dps




class ExecutableTask(object, metaclass=ExecutableTaskMeta):
    
    def __init__(self, input_data_products=None, context=None, **kwargs):                

        self._timing = {}
        self.input_data_products = input_data_products

        # Set parameters.
        self._parameters = {}
        missing_required_parameters = []
        for key, parameter in self.__class__.__dict__.items():
            if not isinstance(parameter, Parameter): continue
            if key in kwargs:
                # We expected this parameter and we have it.
                value = kwargs[key]
                default = False
            else:
                # Check for default value.
                try:
                    value = parameter.default
                except AttributeError:
                    # No default value.
                    missing_required_parameters.append(key)
                    continue
                else:
                    default = True
            
            bundled = parameter.bundled
            self._parameters[key] = (parameter, value, bundled, default)
            setattr(self, key, value)

        if missing_required_parameters:
            M = len(missing_required_parameters)
            raise TypeError(f"__init__() missing {M} required parameter{'s' if M > 1 else ''}: '{', '.join(missing_required_parameters)}'")

        # Check for unexpected parameters.
        unexpected_parameters = set(kwargs.keys()) - set(self._parameters.keys())
        if unexpected_parameters:
            raise TypeError(f"__init__() got unexpected keyword argument{'s' if len(unexpected_parameters) > 1 else ''}: '{', '.join(unexpected_parameters)}'")

        # Set the executable context.
        self.context = context or {}


    def infer_task_bundle_size(self):

        # Figure out how many implied tasks there could be in this bundle.
        if self.input_data_products is not None:
            # List-ify if needed.
            try:
                _ = (item for item in input_data_products)
            except TypeError:
                input_data_products = [input_data_products]

            N = len(input_data_products)

        else:
            Ns = []
            for key, parameter in self.__class__.__dict__.items():
                if isinstance(parameter, Parameter) and not parameter.bundled:
                    value = getattr(self, key)
                    if isinstance(value, (list, tuple, set, np.ndarray)):
                        Ns.append(len(value))                        
            if len(Ns) == 0:
                N = 1
            else:
                N = max(Ns)
        return N


    def iterable(self):
        # TODO: Hook this in with per-task timing for execution!
        try:
            execution_context = inspect.stack()[1].function
        except:
            execution_context = None
            log.warning(f"Cannot infer execution context! {inspect.stack()[1]}")

        times = [time()]
        for item in self.context["iterable"]:
            yield item
            times.append(time())

        task_times = np.diff(times)

        key = f"time_{execution_context}_task"
        self._timing.setdefault(key, np.zeros_like(task_times))
        self._timing[key] += task_times
        log.debug(f"Recorded {execution_context} times of {task_times}")


    def get_or_create_context(self, force=False):
        if len(self.context) > 0:
            return self.context

        module = inspect.getmodule(self)
        name = f"{module.__name__}.{self.__class__.__name__}"
        
        context = {
            "input_data_products": get_or_create_data_products(self.input_data_products),
            "tasks": [],
            "bundle": None,
            "iterable": []
        }

        for i in range(self.infer_task_bundle_size()):
            try:
                data_products = self.input_data_products[i]
            except:
                data_products = None
        
            parameters = {}
            for k, (parameter, value, bundled, default) in self._parameters.items():
                if bundled:
                    parameters[k] = value
                else:
                    try:
                        parameters[k] = value[i]
                    except:
                        parameters[k] = value

            task = Task.create(
                name=name,
                parameters=parameters,
                version=__version__,
            )
            for data_product in flatten(data_products):
                TaskInputDataProducts.create(task=task, data_product=data_product)

            context["tasks"].append(task)
            context["iterable"].append((task, data_products, parameters))

        N = i + 1
        if N > 1 or force:
            # Create task bundle.
            context["bundle"] = bundle = Bundle.create()
            for task in context["tasks"]:
                TaskBundle.create(task=task, bundle=bundle)

        return context


    def _update_task_timings(self):

        N = len(self.context["tasks"])

        time_pre_execute_task = self._timing.get("time_pre_execute_task", np.zeros(N))
        time_execute_task = self._timing.get("time_execute_task", np.zeros(N))
        time_post_execute_task = self._timing.get("time_post_execute_task", np.zeros(N))

        time_pre_execute_bundle = self._timing["actual_time_pre_execute"] - np.sum(time_pre_execute_task)
        time_pre_execute = time_pre_execute_bundle / N + time_pre_execute_task

        time_execute_bundle = self._timing["actual_time_execute"] - np.sum(time_execute_task)
        time_execute = time_execute_bundle / N + time_execute_task

        time_post_execute_bundle = self._timing["actual_time_post_execute"] - np.sum(time_post_execute_task)
        time_post_execute = time_post_execute_bundle / N + time_post_execute_task

        time_total = time_pre_execute + time_execute + time_post_execute

        for i, task in enumerate(self.context["tasks"]):
            kwds = dict(
                time_total=time_total[i],
                time_pre_execute=time_pre_execute[i],
                time_execute=time_execute[i],
                time_post_execute=time_post_execute[i],

                
                time_pre_execute_task=time_pre_execute_task[i],
                time_execute_task=time_execute_task[i],
                time_post_execute_task=time_post_execute_task[i],

                time_pre_execute_bundle=time_pre_execute_bundle,
                time_execute_bundle=time_execute_bundle,
                time_post_execute_bundle=time_post_execute_bundle,                
            )
            r = (
                Task.update(**kwds)
                    .where(Task.pk == task.pk)
                    .execute()
            )

        return None


    @abc.abstractmethod
    def execute(self, *args, **kwargs):
        pass

    @classmethod
    def _execute_decorator(cls, function):
        def do_pre_post(self):
            log.debug(f"Decorating pre/post_execute {self} {self.context}")

            # Do pre-execution.
            t_init = time()
            try:
                pre_execute = self.pre_execute()
            except:
                log.exception(f"Pre-execution failed for {self}:")
                raise
            self._timing["actual_time_pre_execute"] = time() - t_init
                
            # Create tasks in database if they weren't already given.
            if len(self.context) == 0:
                self.context.update(self.get_or_create_context())
            
            # Update context with pre execution.
            if pre_execute is not None:
                self.context.update(pre_execute)

            t_init = time()
            try:
                self.result = function(self)
            except:
                log.exception(f"Execution failed for {self}:")
                # TODO: Mark tasks as failed / delete them?
                raise

            self._timing["actual_time_execute"] = time() - t_init


            t_init = time()
            try:
                # This would check context for the output.
                post_execute = self.post_execute()
            except:
                log.exception(f"Post-execution failed for {self}:")
                raise
                
            self._timing["actual_time_post_execute"] = time() - t_init

            self._update_task_timings()

            return self.result

        return do_pre_post
        

    def pre_execute(self, **kwargs):
        pass

    def post_execute(self, **kwargs):
        pass


