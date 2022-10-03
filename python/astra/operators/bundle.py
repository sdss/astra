import json
import numpy as np
from zoneinfo import available_timezones
from peewee import fn
from airflow.sensors.base import BaseSensorOperator
from airflow.models.baseoperator import BaseOperator
from airflow.exceptions import AirflowFailException
from joblib import Parallel, delayed, parallel_backend

from astra import log, __version__
from astra.database.astradb import Task, Bundle, TaskBundle
from astra.operators.utils import get_or_create_bundle, to_callable, add_to_bundle

def f(task_id):
    try:
        Task.get(task_id).instance().execute()
    except:
        log.exception(f"Exception in executing task id {task_id}:")

class BundleExecutor(BaseOperator):
    
    template_fields = ("bundle", )

    """
    Execute a bundle on the head node. 
    
    This is useful if the bundle has low overheads (e.g., low time to load a model, but many tasks to run).
    This will create an instance of every task in the bundle and execute the task instances in parallel,
    instead of the alternative of creating a single instance of the bundle and executing the bundle instance.
    If you need to execute many bundles, or a bundle with significant overheads, then you should use a SlurmOperator.
    """ 

    def __init__(
        self,
        bundle,
        n_jobs=1,
        backend="threading",
        **kwargs,
    ) -> None:
        super(BundleExecutor, self).__init__(**kwargs)
        self.bundle = bundle
        self.n_jobs = n_jobs
        self.backend = backend
        available_backends = ("loky", "threading", "multiprocessing")
        if backend not in available_backends:
            raise ValueError(f"Backend must be one of '{' '.join(available_backends)}', not '{backend}'")
        return None


    def execute(self, context):
        if isinstance(self.bundle, str):
            bundle_ids = json.loads(self.bundle)
            if isinstance(bundle_ids, (tuple, list)):
                bundle_ids = list(map(int, bundle_ids))
            else:
                bundle_ids = [int(bundle_ids)]
        else:
            bundle_ids = [int(self.bundle)]
        
        self.n_jobs = int(self.n_jobs)

        log.info(f"Executing bundle(s) {bundle_ids} with n_jobs {self.n_jobs} and backend {self.backend}")

        # If they are embarrasingly parallel with low overhead, then this might be helpful
        if self.n_jobs > 1:
            task_ids = (
                TaskBundle
                .select(TaskBundle.task_id)
                .where(TaskBundle.bundle_id.in_(bundle_ids))
                .tuples()
            )            
            with parallel_backend(self.backend, n_jobs=self.n_jobs):
                Parallel()(delayed(f)(task_id) for task_id, in task_ids)
        else:
            if len(bundle_ids) > 1:
                raise ValueError(f"Cannot execute multiple bundles with n_jobs=1")
            Bundle.get(bundle_ids[0]).instance().execute()
        
        return None


class BundleCreator(BaseOperator):

    template_fields = ("executable_task_name", "input_data_products", "parameters")

    def __init__(
        self,
        executable_task_name,
        input_data_products,
        parameters,
        use_existing_bundle=False,
        num_bundles=1, 
        **kwargs
    ) -> None:
        super(BundleCreator, self).__init__(**kwargs)
        self.executable_task_name = executable_task_name
        self.input_data_products = input_data_products
        self.parameters = parameters
        self.use_existing_bundle = use_existing_bundle
        self.num_bundles = num_bundles

        if not self.use_existing_bundle and self.num_bundles < 1:
            raise ValueError(f"If not using existing bundle, num_bundles must be >= 1, not {self.num_bundles}")
        
        if self.use_existing_bundle and self.num_bundles > 1:
            log.warn("Using existing bundle, so ignoring num_bundles keyword")
        return None

    def execute(self, context):
        from astra.database.astradb import Task, Bundle, TaskBundle, TaskInputDataProducts, database

        log.debug(f"Loading executable task: {self.executable_task_name}")
        executable_class = to_callable(self.executable_task_name)
        
        # Resolve input data products
        if isinstance(self.input_data_products, str):
            self.input_data_products = json.loads(self.input_data_products)
        
        # Get parsed parameters
        bundle_size, parsed_parameters = executable_class.parse_parameters(**self.parameters)
        parameters = {k: v for k, (p, v, *_) in parsed_parameters.items()}
        
        # Create a task per data product.
        tasks = [
            Task(name=self.executable_task_name, parameters=parameters, version=__version__)
            for _ in range(len(self.input_data_products))
        ]
        log.info(f"Creating tasks in bulk")
        with database.atomic():
            Task.bulk_create(tasks)
            TaskInputDataProducts.insert_many([
                {"task_id": task.id, "data_product_id": idp } for task, idp in zip(tasks, self.input_data_products)
            ]).execute()

        if self.use_existing_bundle:
            bundle_id = get_or_create_bundle(
                self.executable_task_name,
                parameters,
                parsed=True
            )
            log.info(f"Using existing bundle {bundle_id}. Adding tasks to bundle.")
            add_to_bundle(bundle_id, [task.id for task in tasks])
            return bundle_id

        else:
            bundles = [Bundle() for _ in range(self.num_bundles)]
            with database.atomic():
                Bundle.bulk_create(bundles)

            log.info(f"Created task bundles {bundles}")
            N_tasks = len(tasks)
            N_tasks_per_bundle = int(np.ceil(N_tasks / self.num_bundles))
            with database.atomic():
                (
                    TaskBundle.insert_many(
                        [
                            { "task_id": task.id, "bundle_id": bundles[i // N_tasks_per_bundle].id } for i, task in enumerate(tasks)
                        ]
                    ).execute()
                )
            log.info(f"Done.")

            return [b.id for b in bundles]


class BundleSensor(BaseSensorOperator):

    """Wait until all tasks in a bundle have completed or failed."""

    template_fields = ("bundle_id", )

    def __init__(
        self,
        bundle_id,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.bundle_id = bundle_id
        return None


    def poke(self, context):

        from astra.database.astradb import Task, TaskBundle, Status

        bundle_id = int(self.bundle_id)

        any_with_created_state = (
            Task
            .select()
            .join(TaskBundle)
            .where(
                (TaskBundle.bundle_id == bundle_id)
            &   (Task.status == Status.get(description="created"))
            )
            .exists()
        )
        if any_with_created_state:
            log.info(f"At least one task in bundle {bundle_id} has created state")
            return False
        
        # If all tasks are completed, return success.
        counts = (
            Task
            .select(
                Task.status_id,
                fn.COUNT(Task.status_id)
            )
            .join(TaskBundle)
            .where(TaskBundle.bundle_id == bundle_id)
            .group_by(Task.status_id)
            .tuples()
        )
        log.info(f"In bundle {bundle_id} there are: ")
        counts = { status_id: count for status_id, count in counts }
        for status_id, count in counts.items():
            log.info(f"\t{count} tasks with status {status_id}: {Status.get_by_id(status_id).description}")
        
        if len(counts) == 1 and 5 in counts:
            log.info(f"All tasks in bundle {bundle_id} are completed")
            return True

        # If some tasks have failed, raise some exception.
        failed_status_ids = (6, 7, 8)
        if set(failed_status_ids).intersection(list(counts.keys())):
            raise AirflowFailException(f"Bundle {bundle_id} has failed tasks")

        
