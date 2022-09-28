import json
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
        self.bundle = int(self.bundle)
        self.n_jobs = int(self.n_jobs)

        log.info(f"Executing bundle {self.bundle} with n_jobs {self.n_jobs} and backend {self.backend}")

        # If they are embarrasingly parallel with low overhead, then this might be helpful
        if self.n_jobs > 1:
            # Should we split this into n_jobs bundles and execute them in parallel?
            print(f"#TODO Andy you should split this bundle into X n_jobs and execute them in parallel")
            task_ids = (
                TaskBundle
                .select(TaskBundle.task_id)
                .where(TaskBundle.bundle_id == self.bundle)
            )            
            with parallel_backend(self.backend, n_jobs=self.n_jobs):
                Parallel()(delayed(f)(task_id) for task_id in task_ids)
        else:
            Bundle.get(self.bundle).instance().execute()
        
        return None


class BundleCreator(BaseOperator):

    template_fields = ("executable_task_name", "input_data_products", "parameters")

    def __init__(
        self,
        executable_task_name,
        input_data_products,
        parameters,
        use_existing_bundle=False,
        **kwargs
    ) -> None:
        super(BundleCreator, self).__init__(**kwargs)
        self.executable_task_name = executable_task_name
        self.input_data_products = input_data_products
        self.parameters = parameters
        self.use_existing_bundle = use_existing_bundle
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
        else:
            bundle_id = Bundle.create().id
        
        log.info(f"Using bundle {bundle_id}. Adding tasks to bundle.")

        add_to_bundle(bundle_id, [task.id for task in tasks])
        log.info(f"Done")
        return bundle_id


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

        
