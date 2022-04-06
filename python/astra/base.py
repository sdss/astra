import abc
import os
import inspect
import json
import numpy as np
from time import time
from astra import (log, __version__)
from astra.database.astradb import (database, DataProduct, Task, Status, TaskInputDataProducts, Bundle, TaskBundle)

class Parameter:
    def __init__(self, name, bundled=False, **kwargs):
        self.name = name
        self.bundled = bundled
        if "default" in kwargs:
            self.default = kwargs.get("default")
        

class ExecutableTaskMeta(abc.ABCMeta):
    def __new__(cls, *args):
        new_class = super(ExecutableTaskMeta, cls).__new__(cls, *args)
        new_class.pre_execute = new_class._pre_execute_decorator(new_class.pre_execute)
        new_class.execute = new_class._execute_decorator(new_class.execute)
        new_class.post_execute = new_class._post_execute_decorator(new_class.post_execute)
        return new_class


def get_or_create_data_products(iterable):
    if iterable is None:
        return None

    if isinstance(iterable, str):
        try:
            iterable = json.loads(iterable)
        except:
            pass

    if isinstance(iterable, DataProduct):
        iterable = [iterable]
        
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
                dp = DataProduct.get(id=int(dp))
            except:
                raise TypeError(f"Unknown input data product type ({type(dp)}): {dp}")
            else:
                dps.append(dp)    
    return dps



    

class ExecutableTask(object, metaclass=ExecutableTaskMeta):
    
    @classmethod
    def get_defaults(cls):
        defaults = {}
        for key, parameter in cls.__dict__.items():
            if isinstance(parameter, Parameter):
                try:
                    defaults[key] = parameter.default
                except:
                    continue
        return defaults
        

    @classmethod
    def parse_parameters(cls, **kwargs):
        parameters = {}
        missing_required_parameters = []
        for key, parameter in cls.__dict__.items():
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
            parameters[key] = (parameter, value, bundled, default)
        
        if missing_required_parameters:
            M = len(missing_required_parameters)
            raise TypeError(f"__init__() missing {M} required parameter{'s' if M > 1 else ''}: '{', '.join(missing_required_parameters)}'")

        # Check for unexpected parameters.
        unexpected_parameters = set(kwargs.keys()) - set(parameters.keys())
        if unexpected_parameters:
            raise TypeError(f"__init__() got unexpected keyword argument{'s' if len(unexpected_parameters) > 1 else ''}: '{', '.join(unexpected_parameters)}'")
        return parameters


    def __init__(self, input_data_products=None, context=None, bundle_size=1, **kwargs):
        self._timing = {}
        self.bundle_size = bundle_size
        self.input_data_products = input_data_products
        if isinstance(input_data_products, str):
            try:
                self.input_data_products = json.loads(input_data_products)
            except:
                None

        # Set parameters.
        self._parameters = self.parse_parameters(**kwargs)
        for parameter_name, (parameter, value, bundled, default) in self._parameters.items():
            setattr(self, parameter_name, value)

        # Set the executable context.
        self.context = context or {}

    '''
    def infer_task_bundle_size(self):

        # Figure out how many implied tasks there could be in this bundle.
        if self.input_data_products is not None:
            # List-ify if needed.
            try:
                _ = (item for item in self.input_data_products)
            except TypeError:
                N = 1
            else:
                N = len(self.input_data_products)

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
    '''


    def iterable(self):
        for level in inspect.stack():
            if level.function in ("pre_execute", "execute", "post_execute"):
                execution_context = level.function
                break
        else:
            execution_context = None
            log.warning(f"Cannot infer execution context {inspect.stack}")

        self._timing["iterable"] = [time()]
        for item in self.context["iterable"]:
            try:
                yield item
            except:
                log.exception(f"Exception in {execution_context} for {item}")
                raise
            self._timing["iterable"].append(time())

        task_times = np.diff(self._timing["iterable"])

        key = f"time_{execution_context}_task"
        self._timing.setdefault(key, np.zeros_like(task_times))
        self._timing[key] += task_times


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
        from astra.utils import flatten
        for i in range(self.bundle_size):
            try:
                data_products = context["input_data_products"][i]
            except:
                data_products = None
        
            parameters = {}
            for k, (parameter, value, bundled, default) in self._parameters.items():
                if bundled:
                    parameters[k] = value
                else:
                    # Only apply a parameter per-task if it's list-like and has the same length as the bundle.
                    if (self.bundle_size > 1 
                        and isinstance(value, (list, tuple, set, np.ndarray)) 
                        and len(value) == self.bundle_size
                    ):
                        parameters[k] = value[i]
                    else:
                        parameters[k] = value

            with database.atomic() as txn: 
                task = Task.create(
                    name=name,
                    parameters=parameters,
                    version=__version__,
                )
                
                for data_product in flatten(data_products):
                    TaskInputDataProducts.create(task=task, data_product=data_product)

            context["tasks"].append(task)
            context["iterable"].append((task, data_products, parameters))

        if force or self.bundle_size > 1:
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

        log.debug(f"Updating task timings")
        
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
                    .where(Task.id == task.id)
                    .execute()
            )

        return None


    @abc.abstractmethod
    def pre_execute(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def execute(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def post_execute(self, *args, **kwargs):
        pass


    def update_status(self, description, items=None):
        status = Status.get(description=description)
        N = 0
        if items is None:    
            items = []
            try:
                items.extend(self.context["tasks"])
            except:
                None
            try:
                items.append(self.context["bundle"])
            except:
                None

        with database.atomic():
            for item in items:
                if item is None: continue
                model = item.__class__
                N += (
                    model.update(status=status)
                        .where(model.id == item.id)
                        .execute()
                )
        return N


    @classmethod
    def _pre_execute_decorator(cls, function):
        def do_pre_execution(self):
            log.debug(f"Decorating pre-execute for {self}")
            
            # Create tasks in database if they weren't already given.
            if len(self.context) == 0:
                self.context.update(self.get_or_create_context())

            # Set tasks as running.
            #self.update_status("running")

            # Do pre-execution.
            t_init = time()
            try:
                pre_execute = function(self)
            except:
                log.exception(f"Pre-execution failed for {self}:")
                #self.update_status("failed-pre-execution")
                raise
            self._timing["actual_time_pre_execute"] = time() - t_init
                
            # Update context with pre execution.
            if pre_execute is not None:
                self.context.update(pre_execute)
            return pre_execute
        return do_pre_execution       


    @classmethod
    def _execute_decorator(cls, function):
        def do_pre_post(self):
            
            self.pre_execute()

            t_init = time()
            try:
                self.result = function(self)
            except:
                # Try to see how far we got.
                if "iterable" not in self._timing:
                    log.exception(f"Execution failed for {self} before we could run first task {self.context['tasks'][0]}:")
                else:
                    i = max(0, len(self._timing["iterable"]) - 2)
                    responsible_task = self.context["tasks"][i]
                    log.exception(f"Execution failed for {self}, probably on task {responsible_task}:")
                    #self.update_status("failed-execution")

                raise
            self._timing["actual_time_execute"] = time() - t_init

            self.post_execute()

            return self.result

        return do_pre_post
     


    @classmethod
    def _post_execute_decorator(cls, function):
        def do_post_execution(self):

            log.debug(f"Post-processing for {self}")

            t_init = time()
            try:
                # This would check context for the output.
                post_execute = function(self)
            except:
                log.exception(f"Post-execution failed for {self}:")
                #self.update_status("failed-post-execution")
                raise
                
            self._timing["actual_time_post_execute"] = time() - t_init

            log.debug(f"Post-processing finished.")
            log.debug(f"Updating task timings..")
            self._update_task_timings()
            log.debug(f"Task timings updated.")

            # Set this task / bundle as finished.
            #self.update_status("completed")
            return post_execute
        return do_post_execution     
