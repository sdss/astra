from email.utils import decode_params
import os
import importlib
import json
from re import A
from typing import OrderedDict
from astropy.io import fits
from airflow.models.baseoperator import BaseOperator
from airflow.exceptions import (AirflowFailException, AirflowSkipException)

from peewee import fn, JOIN
from astra.database.astradb import database, TaskOutput, DataProduct, Task, TaskInputDataProducts, Bundle, TaskBundle, Status
from astra.utils import flatten, deserialize
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
            bundle_size, parsed_parameters = executable_class.parse_parameters(**parameters)
            parameters = {k: v for k, (p, v, *_) in parsed_parameters.items() }

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

    return bundle.id



def create_tasks(
        executable_task_name, 
        input_data_products, 
        parameters
    ):

    executable_class = to_callable(executable_task_name)
    if isinstance(input_data_products, str):
        input_data_products = json.loads(input_data_products)

    task_ids = []
    with database.atomic() as txn:
        for i, idp in enumerate(input_data_products):
            bundle_size, parsed_parameters = executable_class.parse_parameters(**parameters)
            parameters = {k: v for k, (p, v, *_) in parsed_parameters.items() }
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
            task_ids.append(task.id)
    
    return task_ids


def get_or_create_bundle(executable_task_name, parameters, status="created"):
    """
    Get an existing bundle that only contains tasks of the given name and bundle parameters
    and has the same status, or create one.
    
    :param executable_task_name:
    
    :param parameters:

    :param status: [optional]
        The status required for any existing bundle.
    """
    
    if isinstance(status, str):
        status = Status.get(description=status)

    # May need to fill in parameters with default values.
    executable_class = to_callable(executable_task_name)    
    bundle_size, parsed_parameters = executable_class.parse_parameters(**parameters)
    parameters = { k: v for k, (p, v, *_) in parsed_parameters.items() }

    q = (
        Bundle.select()
              .join(TaskBundle)
              .join(Task)
              .where(
                  (Task.name == executable_task_name) &
                  (Task.parameters == parameters) & 
                  (Bundle.status == status)
              )
              .group_by(Bundle)
    )

    bundle = q.first() or Bundle.create()
    return bundle.id


def add_to_bundle(bundle_id, task_ids):

    if isinstance(task_ids, str):
        task_ids = json.loads(task_ids)
    
    for task_id in task_ids:
        TaskBundle.create(
            bundle_id=int(bundle_id),
            task_id=int(task_id)
        )
        log.debug(f"Assigned task {task_id} to bundle {bundle_id}")

    return None





from airflow.sensors.base import BaseSensorOperator

class BundleStatusSensor(BaseSensorOperator):

    template_fields = ("bundle", "wait_for_status")

    def __init__(
        self,
        bundle,
        wait_for_status,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.bundle = bundle
        self.wait_for_status = wait_for_status

    def poke(self, context):
        # First check that we don't have too many jobs submitted/running already.

        bundle = Bundle.get_by_id(int(self.bundle))
        return (bundle.status.description != self.wait_for_status)




def check_task_outputs(tasks):
    tasks = deserialize(tasks, Task)

    success, error, total = (0, 0, 0)
    for task in tasks:
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


def check_bundle_outputs(bundles, raise_on_error=True):
    
    bundles = deserialize(bundles, Bundle)

    total, error = (0, 0)
    for bundle in bundles:
        log.info(f"Checking all tasks in bundle {bundle} have outputs")

        n_tasks = bundle.count_tasks()
        log.info(f"  There are {n_tasks} tasks in this bundle.")

        q_no_outputs = (
            Task.select(Task.id)
                .join(TaskBundle)
                .join(Bundle)
                .where(Bundle.id == bundle.id)
                .switch(Task)
                .join(TaskOutput, JOIN.LEFT_OUTER)
                .group_by(Task)
                .having(fn.count(TaskOutput) == 0)
                .tuples()
        )
        n_no_outputs = q_no_outputs.count()
        log.info(f"  There are {n_no_outputs} tasks in this bundle without outputs.")
        for task_id, in q_no_outputs:
            log.warning(f"    Task {task_id} has no outputs.")

        total += n_tasks
        error += n_no_outputs
    
    if raise_on_error and error > 0:
        raise RuntimeError(f"There are tasks without outputs.")
    return None


import pendulum

def skip_if_backfill(dag, next_execution_date, **kwargs):
    """
    Raise an AirflowSkipException if there are more recent DAG executions to take place.
    """
    run_dates = dag.get_run_dates(start_date=next_execution_date + pendulum.duration(seconds=1))
    if run_dates:
        raise AirflowSkipException(f"There are more recent DAG executions to take place: {run_dates}")
    
    return None

