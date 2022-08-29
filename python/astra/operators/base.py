import json
import os
from airflow.compat.functools import cached_property
from airflow.hooks.subprocess import SubprocessHook
from airflow.models.baseoperator import BaseOperator
from airflow.utils.operator_helpers import context_to_airflow_vars

from astra import log
from astra.utils import deserialize, flatten
from astra.operators.utils import to_callable
from astra.database.astradb import Task


class AstraOperator(BaseOperator):

    template_fields = ("task_parameters",)

    def __init__(
        self, task_name, task_parameters=None, return_id_kind="task", **kwargs
    ) -> None:
        super(AstraOperator, self).__init__(**kwargs)
        self.task_name = task_name
        self.task_parameters = task_parameters or {}
        self.return_id_kind = f"{return_id_kind}".lower()
        if self.return_id_kind not in ("task", "data_product"):
            raise ValueError(f"return_id_kind must be either `task` or `data_product`")

    def execute(self, context):
        log.info(
            f"Creating task {self.task_name} with task_parameters {self.task_parameters}"
        )
        executable_class = to_callable(self.task_name)
        task = executable_class(**self.task_parameters)

        log.info(f"Executing")
        task.execute()
        log.info(f"Done")

        if self.return_id_kind == "task":
            outputs = [task.id for task in task.context["tasks"]]
        elif self.return_id_kind == "data_product":
            outputs = []
            for t in task.context["tasks"]:
                for dp in t.output_data_products:
                    outputs.append(dp.id)

        return outputs


class TaskExecutor(BaseOperator):

    template_fields = ("execute_task_ids",)

    def __init__(
        self,
        execute_task_ids,
        **kwargs,
    ) -> None:
        super(TaskExecutor, self).__init__(**kwargs)
        self.execute_task_ids = execute_task_ids
        return None

    def execute(self, context):

        tasks = deserialize(self.execute_task_ids, Task)
        N = len(tasks)

        log.info(f"Executing {N} tasks.")

        for i, task in enumerate(tasks, start=1):
            log.info(f"Executing item {i}/{N}: {task}")
            try:
                result = task.instance().execute()
            except:
                log.exception(f"Exception when executing item {task}")
            else:
                log.info(f"Completed task {task}")

        return None
