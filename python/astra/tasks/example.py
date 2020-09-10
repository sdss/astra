

import luigi
import numpy as np
import sqlite3
from luigi.util import inherits, requires
from sqlalchemy import String
from luigi.mock import MockTarget
from luigi.contrib import sqla


class BaseTask(luigi.Task):

    def output(self):
        return MockTarget(self.task_id)
    
    def write_to_database(self, result):
        with self.output().open("w") as out:
            for key, value in result.items():
                out.write(f"{self.task_id}\t{key}\t{value}\n")


class DatabaseTask(sqla.CopyToTable):
    
    columns = [
        (["task_id", String(64)], {"primary_key": True}),
        (["key", String(64)], {"primary_key": True}),
        (["value", String(64)], {})
    ]

    reflect = luigi.BoolParameter(
        default=True,
        significant=False,
    )
    connection_string = luigi.Parameter("sqlite:///temp8.db")
    table = luigi.Parameter(default="task_results")
    



class Mixin:
    foo = luigi.IntParameter()


class AnalysisTask(Mixin, BaseTask):

    def run(self):
        result = {
            "result": np.random.uniform(),
            "attr": np.random.uniform()
        }

        self.write_to_database(result)



@inherits(AnalysisTask)
@requires(AnalysisTask)
class WriteAnalysisTaskResultToDatabase(DatabaseTask):    
    pass







if __name__ == "__main__":

    tasks = [
        WriteAnalysisTaskResultToDatabase(foo=3)
    ]
    luigi.build(
        tasks,
        local_scheduler=True
    )