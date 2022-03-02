import abc
import os
import inspect
import json
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
        print(f"Checking new class {args}")
        new_class = super(ExecutableTaskMeta, cls).__new__(cls, *args)
        # decorate execute() function to do pre/post execute
        new_class.execute = new_class._execute_decorator(new_class.execute)
        return new_class


def get_or_create_data_products(iterable):
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
    
        # List-ify if needed.
        try:
            _ = (item for item in input_data_products)
        except TypeError:
            input_data_products = [input_data_products]

        N = len(input_data_products)

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
            N = len(missing_required_parameters)
            raise TypeError(f"__init__() missing {N} required parameter{'s' if N > 1 else ''}: '{', '.join(missing_required_parameters)}'")

        # Check for unexpected parameters.
        unexpected_parameters = set(kwargs.keys()) - set(self._parameters.keys())
        if unexpected_parameters:
            raise TypeError(f"__init__() got unexpected keyword argument{'s' if len(unexpected_parameters) > 1 else ''}: '{', '.join(unexpected_parameters)}'")

        # Set the executable context.
        self.context = context or {}


    def iterable(self):
        # TODO: Hook this in with per-task timing for execution!
        try:
            timing_execution = inspect.stack()[1]["function"]
        except:
            timing_execution = None

        #from time import time
        #times = [time()]
        for item in self.context["iterable"]:
            yield item
        #    times.append(time())
        #print(times)
        


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
        
        for i, data_products in enumerate(context["input_data_products"]):
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



    @abc.abstractmethod
    def execute(self, *args, **kwargs):
        pass

    @classmethod
    def _execute_decorator(cls, function):
        def do_pre_post(self):
            log.debug(f"Decorating pre/post_execute {self} {self.context}")

            # Do pre-execution.
            try:
                pre_execute = self.pre_execute()
            except:
                log.exception(f"Pre-execution failed for {self}:")
                raise
            
            # Create tasks in database if they weren't already given.
            if len(self.context) == 0:
                self.context.update(self.get_or_create_context())
            # Update context with pre execution.
            # TODO: Put the pre-execute in with self.get_or_create_context()
            if pre_execute is not None:
                self.context.update(pre_execute)


            # TODO: record time here for execution
            try:
                self.result = function(self)
            except:
                log.exception(f"Execution failed for {self}:")
                # TODO: Mark tasks as failed / delete them?
                raise

            # TODO: record time for post-execution
            try:
                # This would check context for the output.
                post_execute = self.post_execute()
            except:
                log.exception(f"Post-execution failed for {self}:")
                raise

            # Put execution timing in database.
            
            return self.result

        return do_pre_post
        

    def pre_execute(self, **kwargs):
        pass

    def post_execute(self, **kwargs):
        pass


