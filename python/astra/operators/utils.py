from email.utils import decode_params
import os
import importlib
from typing import OrderedDict
from astropy.io import fits
from airflow.models.baseoperator import BaseOperator
from airflow.exceptions import (AirflowFailException, AirflowSkipException)

from astra.database.astradb import DataProduct, Task, TaskInputDataProducts, Bundle, TaskBundle
from astra.utils import flatten
from astra import (__version__, log)

def to_callable(string):
    module_name, func_name = string.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)    


def create_task_bundle(executable_task_name, input_data_products, parameters):

    log.info(f"Creating task for {executable_task_name} with {parameters}")
    log.debug(f"Input data products {type(input_data_products)}: {input_data_products}")

    executable_class = to_callable(executable_task_name)

    # Create a big task with all things. 
    executable_task = executable_class(
        input_data_products=input_data_products,
        **parameters
    )

    # Create tasks in database, and task bundle.
    context = executable_task.get_or_create_context(force=True)

    log.info(f"Created task bundle {context['bundle']} with {len(context['tasks'])} tasks")
    log.info(f"Complete context: {context}")

    return context["bundle"].pk


