import abc
import os
import inspect
import json
import numpy as np
from time import time
from collections.abc import Iterable
from astra import (log, __version__)
from astra.database.astradb import (database, DataProduct, Task, Status, TaskInputDataProducts, Bundle, TaskBundle)


class TaskStageTimer:
    def __init__(self, task, stage):
        self.task = task
        self.stage = stage
    
    def __enter__(self):
        self.t_enter = time()
        return self
    
    def __exit__(self, type, value, traceback):
        self.t_exit = time()
        try:
            self.set_internal_timing(self.t_exit - self.t_enter)
        except:
            log.exception(f"Exception in updating task timing for {self.task} during {self.stage}")

        # Handle exception.
        if traceback is not None:
            log.exception(f"Exception in {self.stage} for {self.task}:\n\t\t{type}: {value} {traceback}")
            self.task.update_status(description=f"failed-{self.stage.replace('_', '-')}", safe=True)
        return None


    def set_internal_timing(self, time_actual):
        self.task.context.setdefault("timing", {})

        # The time per task for this stage is filled by the task.iterable()
        time_per_task = np.array(self.task.context["timing"].get(f"time_{self.stage}_per_task", [0]))
        time_bundle = time_actual - sum(time_per_task) # overhead for bundle

        # These should match the database model for Task
        self.task.context["timing"][f"time_{self.stage}_bundle_overhead"] = time_bundle
        self.task.context["timing"][f"time_{self.stage}"] = time_actual
        self.task.context["timing"]["time_total"] = \
            sum([self.task.context["timing"].get(f"time_{s}", 0) for s in ("pre_execute", "execute", "post_execute")])
        

def decorate_pre_post_execute(f):
    stage = f.__name__
    def wrapper(task, *args, **kwargs):
        if not kwargs.pop(f"decorate_{stage}", True):
            return f(task, *args, **kwargs)

        log.debug(f"Decorating {stage} for {task} with {args} {kwargs}")
        # For pre-execute.
        if len(task.context) == 0: # Create context if none given.
            task.context.update(task.get_or_create_context())
        
        with TaskStageTimer(task, stage):
            result = f(task, *args, **kwargs)

        task.context[stage] = result

        return result
    return wrapper        


def decorate_execute(f):
    def wrapper(task, *args, **kwargs):
        
        decorate_pre_execute = kwargs.pop("decorate_pre_execute", True)
        decorate_execute = kwargs.pop("decorate", True)
        decorate_post_execute = kwargs.pop("decorate_post_execute", True)
        raise_decorator_exceptions = kwargs.pop("raise_decorator_exceptions", False)

        if not decorate_execute:
            return f(task, *args, **kwargs)

        # Set status as running.
        task.update_status(description="running", safe=True)
        task.pre_execute(decorate_pre_execute=decorate_pre_execute, **kwargs)

        log.debug(f"Decorating execute for {task} with {args} {kwargs}")
        with TaskStageTimer(task, "execute"):
            result = task.context["execute"] = f(task, *args, **kwargs)
        
        task.post_execute(decorate_post_execute=decorate_post_execute, **kwargs)

        # After post-execute, update the task status and timings.
        task.update_status(description="completed", safe=True)

        default = np.zeros(task.bundle_size)
        try:
            timing = task.context["timing"].copy()
            time_pre_execute_per_task = timing.pop("time_pre_execute_per_task", default)
            time_execute_per_task = timing.pop("time_execute_per_task", default)
            time_post_execute_per_task = timing.pop("time_post_execute_per_task", default)
            
            with database.atomic():
                for i, item in enumerate(task.context["tasks"]):
                    item.time_pre_execute_task = time_pre_execute_per_task[i]
                    item.time_execute_task = time_execute_per_task[i]
                    item.time_post_execute_task = time_post_execute_per_task[i]
                    for k, v in timing.items():
                        setattr(item, k, v)
                    item.save()

        except:
            log.exception(f"Exception occurred while updating timings for task {task}")
            if raise_decorator_exceptions:
                raise
        
        return result

    return wrapper




class Parameter:
    def __init__(self, name=None, bundled=False, **kwargs):
        self.name = name
        self.bundled = bundled
        if "default" in kwargs:
            self.default = kwargs.get("default")


class TupleParameter(Parameter):
    pass

class DictParameter(Parameter):
    pass
 

class TaskInstanceMeta(type):
    def __new__(cls, class_name, bases, attrs):
        # Decorate stages for timing / exception tracking.
        stages = {
            "pre_execute": decorate_pre_post_execute,
            "execute": decorate_execute,
            "post_execute": decorate_pre_post_execute,
        }
        for stage, decorator in stages.items():
            if stage in attrs:
                attrs[stage] = decorator(attrs[stage])
        return type.__new__(cls, class_name, bases, attrs)



class TaskInstance(object, metaclass=TaskInstanceMeta):
    
    @classmethod
    def get_defaults(cls):
        defaults = {}
        for key, parameter in inspect.getmembers(cls):
            if isinstance(parameter, Parameter):
                try:
                    defaults[key] = parameter.default
                except:
                    continue
        return defaults
        

    @classmethod
    def parse_parameters(cls, **kwargs):
        
        iterable_parameter_classes = (TupleParameter, DictParameter)

        missing = []
        parameters = {}
        expected = { k: p for k, p in inspect.getmembers(cls) if isinstance(p, Parameter) }
        for name, parameter in expected.items():
            try:
                value = kwargs[name]
            except KeyError:
                # Check for default value.
                try:
                    value = parameter.default
                except AttributeError:
                    # No default value.
                    missing.append(name)
                    continue
                else:
                    default = True
            else:
                default = False
            
            try:
                length = len(value)
            except:
                length = 1
            
            parameters[name] = (parameter, value, default, length)

        # Check for missing parameters.
        if missing:
            M = len(missing)
            raise TypeError(f"{cls.__name__}.__init__ missing {M} required parameter{'s' if M > 1 else ''}: '{', '.join(missing)}'")
        # Check for unexpected parameters.
        unexpected_parameters = set(kwargs.keys()) - set(parameters.keys())
        if unexpected_parameters:
            raise TypeError(f"__init__() got unexpected keyword argument{'s' if len(unexpected_parameters) > 1 else ''}: '{', '.join(unexpected_parameters)}'")

        relevant_lengths = {}
        for name, (parameter, value, default, length) in parameters.items():
            # All bundled Non-Tuple/-Dict parameters must be single length.
            # (We can't do everything for the user.)
            if (
                parameter.bundled and not isinstance(parameter, iterable_parameter_classes)
                and not isinstance(value, str) and isinstance(value, Iterable)
                and length > 1
            ):
                raise TypeError(
                    f"Bundled parameter '{name}' in {cls} must be single value (not {length}: {type(value)} {value}). "
                    f"Either create separate tasks per bundled parameter, or re-cast the '{name}' parameter "
                    f"as a TupleParameter or DictParameter."
                )

            # Check the lengths of all non-bundled params.
            if not parameter.bundled and not default and not isinstance(parameter, iterable_parameter_classes):
                relevant_lengths[name] = length
        
        lengths = set(relevant_lengths.values()).difference({1})
        L = len(lengths)
        if L == 0:
            bundle_size = 1
        elif L == 1:
            bundle_size, = lengths
        else:
            raise ValueError(
                f"Non-bundled parameters that aren't TupleParameters or DictParameters "
                f"must all have the same length, otherwise the bundling is not explicit. "
                f"Found lengths: {relevant_lengths}"
            )
        
        # Replace the length with a boolean whether we should give value as-is, or index it.
        parsed = {}
        for name, (parameter, value, default, length) in parameters.items():
            indexed = (
                bundle_size > 1 and length == bundle_size 
                and not default
                and not parameter.bundled 
                and not isinstance(parameter, iterable_parameter_classes)
            )
            parsed[name] = (parameter, value, default, length, indexed)
        return (bundle_size, parsed)


    def __init__(self, input_data_products=None, context=None, **kwargs):
        self.context = context or {} # Set the execution context.
        
        # Set parameters.
        self.bundle_size, self._parameters = self.parse_parameters(**kwargs)
        for parameter_name, (parameter, value, *_) in self._parameters.items():
            setattr(self, parameter_name, value)

        self.input_data_products = input_data_products
        if isinstance(input_data_products, str):
            try:
                self.input_data_products = json.loads(input_data_products)
            except:
                None


    def iterable(self, stage=None):
        if stage is None:
            for level in inspect.stack():
                if level.function in ("pre_execute", "execute", "post_execute"):
                    stage = level.function
                    break
            else:
                log.warning(f"Cannot infer execution context; no timing information will be recorded.")

        if stage is not None:
            key = f"time_{stage}_per_task"
            self.context.setdefault("timing", {})
            self.context["timing"].setdefault(key, [])

        # Create context just-in-time in case we are calling iterable() outside of pre/post/execute.
        if "iterable" not in self.context:
            self.get_or_create_context()

        t_init = time()
        for i, item in enumerate(self.context["iterable"]):
            yield item
            if stage is not None:
                t_iterable = time() - t_init
                try:
                    self.context["timing"][key][i] += t_iterable
                except IndexError: # array is not long enough, we're in append mode
                    self.context["timing"][key].append(t_iterable)
                finally:
                    t_init = time()
        # fin

    
    def get_or_create_context(self):
        if "iterable" in self.context: # TODO: think of a better way to do this
            return self.context

        module = inspect.getmodule(self)
        name = f"{module.__name__}.{self.__class__.__name__}"
        
        context = {
            "input_data_products": get_or_create_data_products(self.input_data_products),
            "tasks": [],
            "bundle": None,
            "iterable": []
        }

        for i in range(self.bundle_size):
            #try:
            #    data_products = context["input_data_products"][i]
            #except:
            #    data_products = None
            data_products = None

            parameters = {}
            for name, (parameter, value, bundled, length, indexed) in self._parameters.items():
                parameters[name] = (value[i] if indexed else value)


            with database.atomic() as txn: 
                task = Task.create(
                    name=name,
                    parameters=parameters,
                    version=__version__,
                )
                
                #for data_product in flatten(data_products):
                #    TaskInputDataProducts.create(task=task, data_product=data_product)

            context["tasks"].append(task)
            context["iterable"].append((task, data_products, parameters))

        if self.bundle_size > 1:
            context["bundle"] = bundle = Bundle.create()
            for task in context["tasks"]:
                TaskBundle.create(task=task, bundle=bundle)

        self.context.update(context)
        return context

    @abc.abstractmethod
    def pre_execute(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def execute(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def post_execute(self, *args, **kwargs):
        pass

    def update_status(self, description, items=None, safe=False):
        args = (self, description, items)
        if safe:
            try:
                N = _update_task_status(*args)
            except:
                log.exception(f"Exception in updating status of task {self} to '{description}'")
                N = 0
        else:
            N = _update_task_status(*args)

        return N


def _update_task_status(task, description, items):
    status = Status.get(description=description)
    N = 0
    if items is None:    
        items = []
        try:
            items.extend(task.context["tasks"])
        except:
            None
        try:
            items.append(task.context["bundle"])
        except:
            None
    with database.atomic():
        for item in items:
            if item is None: continue
            item.status = status
            item.save()
            N += 1
    return N
        

ExecutableTask = TaskInstance





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
