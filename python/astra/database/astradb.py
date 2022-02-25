import json
from functools import lru_cache
from sdss_access import SDSSPath
from peewee import (SqliteDatabase, AutoField, TextField, ForeignKeyField, DateTimeField, BigIntegerField, FloatField, BooleanField)
from sdssdb.connection import PeeweeDatabaseConnection
from sdssdb.peewee import BaseModel
from astra import (config, log)

# The database config should always be present, but let's not prevent importing the module because it's missing.
_database_config = config.get("astra_database", {})

# If a URL is given, that overrides all other config settings.
_database_url = _database_config.get("url", None)
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

    @property
    def path(self):
        kwds = self.kwargs.copy()
        if "field" in kwds:
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


class Task(AstraBaseModel):
    pk = AutoField()
    name = TextField()
    parameters = JSONField(null=True)

    version = TextField(null=True)

    # We want some times to be recorded:
    # - time taken for common pre-execution time (for all sources)
    # - time taken for this task pre-execution time 
    # - time between preparing task for slurm, and actually submitting to slurm
    # - time between waiting for slurm to get the task, and actually start running the task
    # - time taken for common execution time (for all sources)
    # - time taken for this task execution time
    # - time taken for common post_execution time (for all sources in a bundle)
    # - time taken for this task post_execution time


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


class ExecutionContext(AstraBaseModel):
    pk = AutoField()
    status = TextField()
    meta = JSONField()


class TaskExecutionContext(AstraBaseModel):
    pk = AutoField()
    task = ForeignKeyField(Task)
    execution_context = ForeignKeyField(ExecutionContext)


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
    models = AstraBaseModel.__subclasses__()
    if drop_existing_tables:
        database.drop_tables(models)
    database.create_tables(models)