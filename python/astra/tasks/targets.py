
import os
import collections
import datetime
import json
import luigi
import sqlalchemy
from luigi.contrib import sqla
from luigi import LocalTarget
from luigi.event import Event
from luigi.mock import MockTarget


class BaseDatabaseTarget(luigi.Target):
    # BaseMixin gives us the connection_string parameter.

    _engine_dict = {}
    Connection = collections.namedtuple("Connection", "engine pid")

    def __init__(self, connection_string, task_namespace, task_family, task_id, schema, echo=True):
        self.task_namespace = task_namespace
        self.task_family = task_family
        self.task_id = task_id

        self.schema = [
            sqlalchemy.Column("task_id", sqlalchemy.String(128), primary_key=True),
        ]
        self.schema.extend(schema)
        self.echo = echo
        self.connect_args = {}
        self.connection_string = connection_string
        return None       


    @property
    def engine(self):
        """
        Return an engine instance, creating it if it doesn't exist.

        Recreate the engine connection if it wasn't originally created
        by the current process.
        """
        pid = os.getpid()
        conn = BaseDatabaseTarget._engine_dict.get(self.connection_string)
        if not conn or conn.pid != pid:
            # create and reset connection
            engine = sqlalchemy.create_engine(
                self.connection_string,
                connect_args=self.connect_args,
                echo=self.echo
            )
            BaseDatabaseTarget._engine_dict[self.connection_string] = self.Connection(engine, pid)
        return BaseDatabaseTarget._engine_dict[self.connection_string].engine


    @property
    def __tablename__(self):
        return f"{self.task_namespace}_{self.task_family}"
    

    @property
    def table_bound(self):
        try:
            return self._table_bound
        except AttributeError:
            return self.create_table()


    def create_table(self):
        """
        Create a table if it doesn't exist.
        Use a separate connection since the transaction might have to be reset.
        """
        with self.engine.begin() as con:
            metadata = sqlalchemy.MetaData()
            if not con.dialect.has_table(con, self.__tablename__):
                self._table_bound = sqlalchemy.Table(
                    self.__tablename__, metadata, *self.schema
                )
                metadata.create_all(self.engine)
            else:
                #metadata.reflect(only=[self.results_table], bind=self.engine)
                #self._table_bound = metadata.tables[self.results_table]
                self._table_bound = sqlalchemy.Table(
                    self.__tablename__,
                    metadata,
                    autoload=True,
                    autoload_with=self.engine
                )

        return self._table_bound
        

    def exists(self):
        r = self._read(self.table_bound, columns=[self.table_bound.c.task_id]) 
        return r is not None


    def read(self, as_dict=False):
        return self._read(self.table_bound, as_dict=as_dict)
        

    def _read(self, table, columns=None, as_dict=False):
        columns = columns or [table]
        with self.engine.begin() as connection:
            s = sqlalchemy.select(columns).where(
                table.c.task_id == self.task_id
            ).limit(1)
            row = connection.execute(s).fetchone()
        if as_dict:
            column_names = (column.name for column in table.columns)
            return collections.OrderedDict(zip(column_names, row))
        return row


    def write(self, data):
        exists = self.exists()
        table = self.table_bound
        sanitised_data = {}
        for key, value in data.items():
            # Don't sanitise booleans or date/datetime objects.
            if not isinstance(value, (datetime.datetime, datetime.date, bool)):
                try:
                    value = str(value)
                except:
                    value = json.dumps(value)
                
            sanitised_data[key] = value
    
        with self.engine.begin() as connection:
            if not exists:
                insert = table.insert().values(
                    task_id=self.task_id,
                    **sanitised_data
                )
            else:
                insert = table.update().where(
                    table.c.task_id == self.task_id
                ).values(
                    task_id=self.task_id,
                    **sanitised_data
                )
            connection.execute(insert)
        return None



class DatabaseTarget(BaseDatabaseTarget):

    """ 
    A database target for outputs of task results. 
    
    This class should be sub-classed, where the sub-class has the attribute `results_schema` that is a list containing the
    table columns for the results table. For example:

    ```
    results_schema = [
        sqlalchemy.Column("effective_temperature", sqlalchemy.Float()),
        sqlalchemy.Column("surface_gravity", sqlalchemy.Float())
    ]
    ```

    The `task_id` of the task supplied will be added as a column by default.
    """

    def __init__(self, task, echo=False, only_significant=True):
        """
        A database target for outputs of task results.

        :param task:
            The task that this output will be the target for. This is necessary to reference the task ID, and to generate the table
            schema for the task parameters.

        :param results_schema: [optional]
            Optionally provide the results schema here, instead of setting it through self.

        :param echo: [optional]
            Echo the SQL queries that are supplied (default: False).
        
        :param only_significant: [optional]
            When storing the parameter values of the task in a database, only store the significant parameters (default: True).
        """
        
        self.task = task
        self.only_significant = only_significant

        schema = generate_parameter_schema(task, only_significant)

        for key, value in self.__class__.__dict__.items():
            if isinstance(value, sqlalchemy.Column):
                schema.append(value)

        super(DatabaseTarget, self).__init__(
            task.connection_string,
            task.task_namespace,
            task.task_family,
            task.task_id,
            schema,
            echo=echo
        )
        return None

    def write(self, data, mark_complete=True):
        # Update with parameter keyword arguments.
        data = data.copy()
        for parameter_name in self.task.get_param_names():
            data[parameter_name] = getattr(self.task, parameter_name)
        
        super(DatabaseTarget, self).write(data)
        if mark_complete:
            self.task.trigger_event(Event.SUCCESS, self.task)
            

def generate_parameter_schema(task, only_significant=True):

    jsonify = lambda _: json.dumps(_)

    mapping = {
        # TODO: Including sanitizers if they are useful in future, but may not be needed.
        luigi.parameter.Parameter: (sqlalchemy.String(1024), None),
        luigi.parameter.OptionalParameter: (sqlalchemy.String(1024), None),
        luigi.parameter.DateParameter: (sqlalchemy.Date(), None),
        luigi.parameter.IntParameter: (sqlalchemy.Integer(), None),
        luigi.parameter.FloatParameter: (sqlalchemy.Float(), None),
        luigi.parameter.BoolParameter: (sqlalchemy.Boolean(), None),
        luigi.parameter.DictParameter: (sqlalchemy.String(1024), jsonify),
        luigi.parameter.ListParameter: (sqlalchemy.String(1024), jsonify),
        luigi.parameter.TupleParameter: (sqlalchemy.String(1024), jsonify)
    }
    parameters_schema = []
    for parameter_name, parameter_type in task.get_params():
        if only_significant and not parameter_type.significant:
            continue
        
        try:
            column_type, sanitize = mapping[parameter_type.__class__]
        except KeyError:
            raise ValueError(f"Cannot find mapping to parameter class {mapping_type.__class__}")
        parameters_schema.append(
            sqlalchemy.Column(
                parameter_name,
                column_type
            )
        )

    return parameters_schema





if __name__ == "__main__":

    from astra.tasks.base import BaseTask
    from sqlalchemy import Column, Integer

    class MyTaskResultTarget(DatabaseTarget):
        
        a = Column("a", Integer)
        b = Column("b", Integer)
        c = Column("c", Integer)
        


    class MyTask(BaseTask):

        param_1 = luigi.FloatParameter()
        param_2 = luigi.IntParameter()
        param_3 = luigi.Parameter()
        
        def output(self):
            return MyTaskResultTarget(self)


        def run(self):
            self.output().write({"a": 5, "b": 3, "c": 2})
            print("Done")


    A = MyTask(param_1=3.5, param_2=4, param_3="what")

    A.run()
    print(A.output().read())
    print(A.output().read(as_dict=True))