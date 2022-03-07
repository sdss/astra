import datetime
import json
import os
from functools import (lru_cache, cached_property)
from sdss_access import SDSSPath
from peewee import (SQL, SqliteDatabase, AutoField, TextField, ForeignKeyField, DateTimeField, BigIntegerField, FloatField, BooleanField)
from sdssdb.connection import PeeweeDatabaseConnection
from sdssdb.peewee import BaseModel
from astra import (config, log)
from astra.utils import flatten
from astra import __version__
from importlib import import_module

# The database config should always be present, but let's not prevent importing the module because it's missing.
_database_config = config.get("astra_database", {})

try:
    # Environment variable overrides all, for testing purposes.
    _database_url = os.environ["ASTRA_DATABASE_URL"]
    if _database_url is not None:
        log.info(f"Using ASTRA_DATABASE_URL enironment variable")
except KeyError:
    _database_url = _database_config.get("url", None)

# If a URL is given, that overrides all other config settings.
if _database_url:
    from playhouse.db_url import connect
    database = connect(_database_url)
    schema = None

else:
    class AstraDatabaseConnection(PeeweeDatabaseConnection):
        dbname = _database_config.get("dbname", None)
        
    database = AstraDatabaseConnection(autoconnect=True)
    schema = _database_config.get("schema", None)

    profile = _database_config.get("profile", None)
    if profile is not None:
        try:
            database.set_profile(profile)
        except AssertionError as e:
            log.exception(e)
            log.warning(f"""
            Database profile '{profile}' set in Astra configuration file, but there is no database 
            profile called '{profile}' found in ~/.config/sdssdb/sdssdb.yml -- it should look like:
            
            {profile}:
                user: [USER]
                host: [HOST]
                port: 5432
                domain: [DOMAIN]

            See https://sdssdb.readthedocs.io/en/stable/intro.html#supported-profiles for more details. 
            If the profile name '{profile}' is incorrect, you can change the 'database' / 'profile' key 
            in ~/.astra/astra.yml
            """)


class AstraBaseModel(BaseModel):
    class Meta:
        database = database
        schema = schema


class JSONField(TextField):
    def db_value(self, value):
        return json.dumps(value)

    def python_value(self, value):
        if value is not None:
            return json.loads(value)
        

@lru_cache
def _lru_sdsspath(release):
    return SDSSPath(release=release)


class Source(AstraBaseModel):
    catalogid = BigIntegerField(primary_key=True)

    # TODO: Include things like Gaia / 2MASS photometry?
    # TODO: Do we even need these two?
    sdssv_target0 = BigIntegerField(null=True)
    sdssv_first_carton_name = TextField(null=True)

    @property
    def data_products(self):
        return (
            DataProduct.select()
                       .join(SourceDataProduct)
                       .join(Source)
                       .where(Source.catalogid == self.catalogid)
        )        


class DataProduct(AstraBaseModel):
    pk = AutoField()
    release = TextField()
    filetype = TextField()
    kwargs = JSONField()

    metadata = JSONField(null=True)

    class Meta:
        indexes = (
            (("release", "filetype", "kwargs"), True),
        )

    @property
    def input_to_tasks(self):
        return (
            Task.select()
                .join(TaskInputDataProducts)
                .join(DataProduct)
                .where(DataProduct.pk == self.pk)
        )

    @cached_property
    def path(self):
        kwds = self.kwargs.copy()
        if "field" in kwds:
            if kwds["field"].startswith(" "):
                log.warning(f"Field name of {self.release} {self.filetype} {self.kwargs} starts with spaces.")
                kwds["field"] = str(kwds["field"]).strip()
        return _lru_sdsspath(self.release).full(self.filetype, **kwds)
        

    @property
    def sources(self):
        return (
            Source.select()
                  .join(SourceDataProduct)
                  .join(DataProduct)
                  .where(DataProduct.pk == self.pk)
        )

# DataProducts and Sources should be a many-to-many relationship.
class SourceDataProduct(AstraBaseModel):
    pk = AutoField()
    source = ForeignKeyField(Source)
    data_product = ForeignKeyField(DataProduct)

    class Meta:
        indexes = (
            (("source", "data_product"), True),
        )


class Output(AstraBaseModel):
    pk = AutoField()
    created = DateTimeField(default=datetime.datetime.now)


class Task(AstraBaseModel):
    pk = AutoField()
    name = TextField()
    parameters = JSONField(null=True)

    version = TextField()

    time_total = FloatField(null=True)
    time_pre_execute = FloatField(null=True)
    time_execute = FloatField(null=True)
    time_post_execute = FloatField(null=True)
    

    time_pre_execute_bundle = FloatField(null=True)
    time_pre_execute_task = FloatField(null=True)

    time_execute_bundle = FloatField(null=True)
    time_execute_task = FloatField(null=True)

    time_post_execute_bundle = FloatField(null=True)
    time_post_execute_task = FloatField(null=True)

    created = DateTimeField(default=datetime.datetime.now)
    completed = DateTimeField(null=True)

    def as_executable(self):
        if self.version != __version__:
            log.warning(f"Task version mismatch for {self}: {self.version} != {__version__}")
            
        module_name, class_name = self.name.rsplit(".", 1)
        module = import_module(module_name)
        executable_class = getattr(module, class_name)

        input_data_products = list(self.input_data_products)

        # We already have context for this task.
        context = {
            "input_data_products": input_data_products,
            "tasks": [self],
            "bundle": None, # TODO: What if this task is part of a bundle,..?
            "iterable": [(self, input_data_products, self.parameters)]
        }
        executable = executable_class(
            input_data_products=input_data_products,
            context=context,
            **self.parameters
        )
        return executable


    @property
    def input_data_products(self):
        return (
            DataProduct.select()
                       .join(TaskInputDataProducts)
                       .join(Task)
                       .where(Task.pk == self.pk)
        )

    @property
    def output_data_products(self):
        return (
            DataProduct.select()
                       .join(TaskOutputDataProducts)
                       .join(Task)
                       .where(Task.pk == self.pk)
        )


    @property
    def outputs(self):
        '''
        q = None
        # Create a compound union query to retrieve all possible outputs for this task.
        o = TaskOutput.get(TaskOutput.task == self)
        for expr, column in o.output.dependencies():
            if column.model != TaskOutput:def
                sq = column.model.select().where(column.model.task == self)
                if q is None:
                    q = sq
                else:
                    q += sq
        # Order by the order they were created.
        return q#.order_by(SQL("output_id").asc())
        '''
        outputs = []
        o = TaskOutput.get(TaskOutput.task == self)
        for expr, column in o.output.dependencies():
            if column.model != TaskOutput:
                outputs.extend(column.model.select().where(column.model.task == self))
        return sorted(outputs, key=lambda x: x.output_id)
    
    def count_outputs(self):
        return TaskOutput.select().where(TaskOutput.task == self).count()


class TaskOutput(AstraBaseModel):
    pk = AutoField()
    task = ForeignKeyField(Task)
    output = ForeignKeyField(Output)


class Bundle(AstraBaseModel):
    pk = AutoField()
    status = TextField(default=0)
    meta = JSONField(null=True)

    @property
    def tasks(self):
        return (
            Task.select()
                .join(TaskBundle)
                .join(Bundle)
                .where(Bundle.pk == self.pk)
        )


    def as_executable(self):

        # Get all the tasks in this bundle.
        tasks = list(
            Task.select()
                .join(TaskBundle)
                .where(TaskBundle.bundle == self)
        )
        input_data_products = [task.input_data_products for task in tasks]

        task = tasks[0]
        if task.version != __version__:
            log.warning(f"Task version mismatch for {self}: {task.version} != {__version__}")
            
        module_name, class_name = task.name.rsplit(".", 1)
        module = import_module(module_name)
        executable_class = getattr(module, class_name)

        context = {
            "input_data_products": input_data_products,
            "tasks": tasks,
            "bundle": self,
            "iterable": [(task, idp, task.parameters) for task, idp in zip(tasks, input_data_products)]
        }

        from astra.base import Parameter
        parameter_names = dict([(k, v.bundled) for k, v in executable_class.__dict__.items() if isinstance(v, Parameter)])

        parameters = {}
        for i, task in enumerate(tasks):
            for p, bundled in parameter_names.items():
                if bundled:
                    if i == 0:
                        parameters[p] = task.parameters[p]
                else:
                    parameters.setdefault(p, [])
                    parameters[p].append(task.parameters[p])
        
        # Keep it simple.
        for p, b in parameter_names.items():
            if not b and len(set(parameters[p])) == 1:
                parameters[p] = parameters[p][0]

        executable = executable_class(
            input_data_products=input_data_products,
            context=context,
            **parameters
        )
        return executable

        


class TaskBundle(AstraBaseModel):
    task = ForeignKeyField(Task)
    bundle = ForeignKeyField(Bundle)


class TaskInputDataProducts(AstraBaseModel):
    pk = AutoField()
    task = ForeignKeyField(Task)
    data_product = ForeignKeyField(DataProduct)
    

class TaskOutputDataProducts(AstraBaseModel):
    pk = AutoField()
    task = ForeignKeyField(Task)
    data_product = ForeignKeyField(DataProduct)




def create_tables(drop_existing_tables=False):
    """ Create all tables for the Astra database. """
    database.connect(reuse_if_open=True)
    models = (Source, DataProduct, SourceDataProduct, Output, Task, TaskOutput, Bundle, TaskBundle, TaskInputDataProducts, TaskOutputDataProducts)
    if drop_existing_tables:
        database.drop_tables(models)
    database.create_tables(models)

