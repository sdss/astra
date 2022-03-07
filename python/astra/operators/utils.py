from email.utils import decode_params
import os
import importlib
import json
from typing import OrderedDict
from astropy.io import fits
from airflow.models.baseoperator import BaseOperator
from airflow.exceptions import (AirflowFailException, AirflowSkipException)

from astra.database.astradb import database, DataProduct, Task, TaskInputDataProducts, Bundle, TaskBundle
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

    if isinstance(input_data_products, str):
        input_data_products = json.loads(input_data_products)
    
    with database.atomic() as txn:
        bundle = Bundle.create()

        for i, idp in enumerate(input_data_products):
            parameters = {k: v for k, (p, v, b, d) in executable_class.parse_parameters(**parameters).items() }

            task = Task.create(
                name=executable_task_name,
                parameters=parameters,
                version=__version__
            )
            for data_product_id in flatten(idp):
                TaskInputDataProducts.create(
                    task=task,
                    data_product_id=data_product_id
                )
            TaskBundle.create(task=task, bundle=bundle)
    
    log.info(f"Created task bundle {bundle} with {i + 1} tasks")

    return bundle.pk



def check_task_bundle_outputs(bundle_pk):
    log.info(f"Checking all tasks in bundle {bundle_pk} have outputs")

    bundle = Bundle.get(pk=int(bundle_pk))

    success, error, total = (0, 0, 0)
    for task in bundle.tasks:
        N = task.count_outputs()
        if N == 0:
            log.error(f"Task {task} has no outputs.")
            error += 1
        else:            
            success += 1
            total += N

    log.info(f"Recorded {success} tasks with a total of {total} outputs and and {error} tasks without any")

    if error > 0:
        raise RuntimeError(f"Some tasks had no outputs.")

    return None
