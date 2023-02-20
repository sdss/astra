import json
from airflow.models.baseoperator import BaseOperator

from astra.utils import log, flatten, to_callable
from astra.database.astradb import DataProduct


class TaskOperator(BaseOperator):

    template_fields = ("task_kwargs", )

    def __init__(self, task_callable, task_kwargs=None, **kwargs):
        super(TaskOperator, self).__init__(**kwargs)
        self.task_callable = task_callable
        self.task_kwargs = task_kwargs or {}
        return None


    def execute(self, context):
        
        task_callable = to_callable(self.task_callable) if isinstance(self.task_callable, str) else self.task_callable
        
        log.info(
            f"Executing task {task_callable} with task_kwargs {self.task_kwargs}"
        )

        # Resolve data_products from identifiers.
        kwargs = self.task_kwargs.copy()
        dp_key = "data_product"
        if dp_key in kwargs:
            if isinstance(kwargs[dp_key], str):
                kwargs[dp_key] = json.loads(kwargs[dp_key])
            if isinstance(kwargs[dp_key], (list, tuple)) and isinstance(kwargs[dp_key][0], int):
                kwargs[dp_key] = DataProduct.select().where(DataProduct.id << kwargs[dp_key])

        log.info(f"Executing")
        results = task_callable(**kwargs)
        log.info(f"Done")

        task_ids = [result.task.id for result in results]
        return task_ids


class GetSourceOperator(BaseOperator):

    template_fields = ("data_product", )

    def __init__(self, data_product, **kwargs):
        super(GetSourceOperator, self).__init__(**kwargs)
        self.data_product = data_product
        return None

    def execute(self, context):
        from astra.database.astradb import Source, DataProduct

        data_product = self.data_product
        if isinstance(data_product, str):
            data_product = flatten(json.loads(data_product))
        print(f"data products: {len(data_product)}")
        
        q = (
            Source
            .select(Source.catalogid)
            .distinct()
            .join(DataProduct)
            .where(
                DataProduct.id << data_product
            )
            .tuples()
        )

        return flatten(q)
