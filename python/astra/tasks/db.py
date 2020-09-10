

import luigi
import numpy as np
import sqlite3
from luigi.util import inherits, requires
import sqlalchemy
from luigi.mock import MockTarget
from luigi.contrib import sqla

from astra.tasks.base import BaseTask


class DatabaseTarget(sqla.SQLAlchemyTarget):

    def __init__(
            self, 
            connection_string, 
            target_table, 
            result_table,
            update_id,
            echo=False,
            connect_args=None
        ):
        super(DatabaseTarget, self).__init__(
            connection_string,
            target_table,
            update_id,
            echo=echo,
            connect_args=connect_args
        )
        self.result_table = result_table
        self.result_table_bound = None
        return None


    def write(self, result):

        self.touch()

        if self.result_table_bound is None:
            self.create_result_table()

        table = self.result_table_bound
        for key, value in result.items():

            with self.engine.begin() as conn:
                s = sqlalchemy.select([table]).where(
                    sqlalchemy.and_(
                        table.c.update_id == self.update_id,
                        table.c.key == key
                    )
                ).limit(1)
                row = conn.execute(s).fetchone()
            
            result_exists = row is not None

            with self.engine.begin() as conn:
                if not result_exists:
                    insert = table.insert().values(
                        update_id=self.update_id,
                        key=key,
                        value=value
                    )
                else:
                    insert = table.update().where(
                        sqlalchemy.and_(
                            table.c.update_id == self.update_id,
                            table.c.key == key
                        )
                    ).values(
                        update_id=self.update_id,
                        key=key,
                        value=value
                    )

                conn.execute(insert)
    

    def create_result_table(self):
        """
        Create a result table if it doesn't exist.

        Using a separate connection since the transaction might have to be reset.
        """
        with self.engine.begin() as con:
            metadata = sqlalchemy.MetaData()
            if not con.dialect.has_table(con, self.result_table):
                self.result_table_bound = sqlalchemy.Table(
                    self.result_table, metadata,
                    sqlalchemy.Column("update_id", sqlalchemy.String(128), primary_key=True),
                    sqlalchemy.Column("key", sqlalchemy.String(128), primary_key=True),
                    sqlalchemy.Column("value", sqlalchemy.String(128))
                )
                metadata.create_all(self.engine)
            
            else:
                metadata.reflect(only=[self.result_table], bind=self.engine)
                self.result_table_bound = metadata.tables[self.result_table]
        


class DatabaseTask(BaseTask):

    connection_string = luigi.Parameter(
        default="sqlite://",
        config_path=dict(section="task_history", name="db_connection"),
        visibility=luigi.parameter.ParameterVisibility.HIDDEN
    )
    task_status_table = luigi.Parameter(
        default="task_status",
        config_path=dict(section="task_history", name="task_status_table"),
        visibility=luigi.parameter.ParameterVisibility.HIDDEN
    )
    task_result_table = luigi.Parameter(
        default="task_results",
        config_path=dict(section="task_history", name="task_result_table"),
        visibility=luigi.parameter.ParameterVisibility.HIDDEN,
        significant=False
    )
    
    def output(self):
        return self.output_database
    
    @property
    def output_database(self):
        return DatabaseTarget(
                self.connection_string,
                self.task_status_table,
                self.task_result_table,
                self.task_id
            )
