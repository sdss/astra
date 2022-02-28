import importlib
from airflow.models.baseoperator import BaseOperator
from sdss_access import SDSSPath
from astra.database.apogee_drpdb import Star, Visit
from astra.database.astradb import (database, DataProduct)
from astra import log
from astra.utils import flatten
from functools import lru_cache


class AstraOperator(BaseOperator):

    def __init__(
        self,
        executable_class,
        parameters=None,
        **kwargs
    ) -> None:
        super(AstraOperator, self).__init__(**kwargs)
        self.executable_class = executable_class
        self.parameters = parameters or {}



    def execute(self, context):
        log.info(f"Executing task {self} with context {context}")

        module_name, class_name = self.executable_class.rsplit(".", 1)
        module = importlib.import_module(module_name)
        executable_class = getattr(module, class_name)

        # Get data products from primary keys immediately upstream.
        task, ti = (context["task"], context["ti"])
        pks = ti.xcom_pull(task_ids=[ut.task_id for ut in task.upstream_list])

        log.info(f"Upstream primary keys: {pks}")

        pks = flatten(pks)

        input_data_products = [DataProduct.get(pk=pk) for pk in pks]

        executable = executable_class(input_data_products, **self.parameters)

        log.info("executing")
        executable.execute()
        log.info("executed")


        