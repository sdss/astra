
import os
import collections
import datetime
import json
import luigi
import sqlalchemy
from luigi.contrib import sqla



class BaseDatabaseTarget(luigi.Target):
    # BaseMixin gives us the connection_string parameter.

    _engine_dict = {}
    Connection = collections.namedtuple("Connection", "engine pid")

    def __init__(self, connection_string, task_namespace, task_family, task_id, parameters_schema, results_schema, echo=True):
        self.task_namespace = task_namespace
        self.task_family = task_family
        self.task_id = task_id

        pk_column = sqlalchemy.Column("task_id", sqlalchemy.String(128), primary_key=True)
        self.results_schema = [pk_column]
        self.results_schema.extend(results_schema)
        self.parameters_schema = [pk_column]
        self.parameters_schema.extend(parameters_schema)
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
    def table_prefix(self):
        return f"{self.task_namespace}_{self.task_family}"
    

    @property
    def parameters_table(self):
        return f"{self.table_prefix}_parameters"


    def create_parameters_table(self):
        """
        Create the parameters table if it doesn't exist.
        Use a separate connection since the transaction might have to be reset.
        """
        with self.engine.begin() as con:
            metadata = sqlalchemy.MetaData()
            if not con.dialect.has_table(con, self.parameters_table):
                self._parameters_table_bound = sqlalchemy.Table(
                    self.parameters_table, metadata, *self.parameters_schema,
                )
                metadata.create_all(self.engine)
            
            else:
                #metadata.reflect(only=[self.parameters_table], bind=self.engine)
                #self._parameters_table_bound = metadata.tables[self.parameters_table]
                self._parameters_table_bound = sqlalchemy.Table(
                    self.parameters_table,
                    metadata,
                    autoload=True,
                    autoload_with=self.engine
                )

        return self._parameters_table_bound


    @property
    def parameters_table_bound(self):
        try:
            return self._parameters_table_bound
        except AttributeError:
            return self.create_parameters_table()


    @property
    def results_table(self):
        return f"{self.table_prefix}_results"


    def create_results_table(self):
        """
        Create a results table if it doesn't exist.
        Use a separate connection since the transaction might have to be reset.
        """
        with self.engine.begin() as con:
            metadata = sqlalchemy.MetaData()
            if not con.dialect.has_table(con, self.results_table):
                self._results_table_bound = sqlalchemy.Table(
                    self.results_table, metadata, *self.results_schema
                )
                metadata.create_all(self.engine)
            else:
                #metadata.reflect(only=[self.results_table], bind=self.engine)
                #self._results_table_bound = metadata.tables[self.results_table]
                self._results_table_bound = sqlalchemy.Table(
                    self.results_table,
                    metadata,
                    autoload=True,
                    autoload_with=self.engine
                )


        return self._results_table_bound

    @property
    def results_table_bound(self):
        try:
            return self._results_table_bound
        except AttributeError:
            return self.create_results_table()


    def exists(self):
        return self.results_exist()


    def results_exist(self):
        return self.read_results() is not None


    def read_results(self, as_dict=False):
        return self._read(self.results_table_bound, as_dict)


    def read_parameters(self, as_dict=False):
        return self._read(self.parameters_table_bound, as_dict)
        

    def _read(self, table, as_dict):
        with self.engine.begin() as connection:
            s = sqlalchemy.select([table]).where(
                table.c.task_id == self.task_id
            ).limit(1)
            row = connection.execute(s).fetchone()
        if as_dict:
            column_names = (column.name for column in table.columns)
            return collections.OrderedDict(zip(column_names, row))
        return row


    def read(self, as_dict=False):
        return self.read_results(as_dict=as_dict)


    def write_results(self, data):
        exists = self.exists()
        table = self.results_table_bound
        with self.engine.begin() as connection:
            if not exists:
                insert = table.insert().values(
                    task_id=self.task_id,
                    **data
                )
            else:
                insert = table.update().where(
                    table.c.task_id == self.task_id
                ).values(
                    task_id=self.task_id,
                    **data
                )
            connection.execute(insert)
        return None


    def write_parameters(self, data):
        exists = self.read_parameters() is not None
        table = self.parameters_table_bound
        sanitised_data = {}
        for key, value in data.items():
            if not isinstance(value, (datetime.datetime, datetime.date)):
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
        

    def write(self, data):
        return self.write_results(data)



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

        :param echo: [optional]
            Echo the SQL queries that are supplied (default: False).
        
        :param only_significant: [optional]
            When storing the parameter values of the task in a database, only store the significant parameters (default: True).
        """
        
        self.task = task
        self.only_significant = only_significant

        self.parameters_schema = generate_parameter_schema(
            task,
            only_significant=only_significant
        )

        super(DatabaseTarget, self).__init__(
            task.connection_string,
            task.task_namespace,
            task.task_family,
            task.task_id,
            self.parameters_schema,
            self.results_schema,
            echo=echo
        )
        return None


    def write_parameters(self):
        data = { 
            k: getattr(self.task, k) for k, pt in self.task.get_params() \
                if (self.only_significant and pt.significant) or not self.only_significant
        }
        return super(DatabaseTarget, self).write_parameters(data)



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


    class MyTaskResultTarget(DatabaseTarget):
        results_schema = [
            sqlalchemy.Column("a", sqlalchemy.Integer),
            sqlalchemy.Column("b", sqlalchemy.Integer),
            sqlalchemy.Column("c", sqlalchemy.Integer),
        ]

    from astra.tasks.base import BaseTask

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
    print(A.output().write_parameters())
    print(A.output().read_parameters())
