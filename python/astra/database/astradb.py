import json
from peewee import AutoField, TextField, ForeignKeyField, DateTimeField
from sdssdb.connection import PeeweeDatabaseConnection
from sdssdb.peewee import BaseModel as _BaseModel
from astra import (config, log)

# The database config should always be present, but let's not prevent importing the module because it's missing.
_database_config = config.get("astra_database", {})

class AstraDatabaseConnection(PeeweeDatabaseConnection):
    dbname = _database_config.get("dbname", None)
    
database = AstraDatabaseConnection(autoconnect=True)

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


class JSONField(TextField):
    def db_value(self, value):
        return json.dumps(value)

    def python_value(self, value):
        if value is not None:
            return json.loads(value)


class BaseModel(_BaseModel):
    class Meta:
        database = database
        schema = _database_config.get("schema", None)
        

class DataProduct(BaseModel):
    pk = AutoField()
    filetype = TextField()
    kwargs = JSONField()

    class Meta:
        indexes = (
            (("filetype", "kwargs"), True),
        )

    @property
    def input_to_tasks(self):
        return (
            Task.select()
                .join(TaskInputDataProducts)
                .join(DataProduct)
                .where(DataProduct.pk == self.pk)
        )


class Task(BaseModel):
    pk = AutoField()
    name = TextField()
    parameters = JSONField(null=True)

    git_hash = TextField(null=True)
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


class ExecutionContext(BaseModel):
    pk = AutoField()
    status = TextField()
    meta = JSONField()



class TaskExecutionContext(BaseModel):
    pk = AutoField()
    task = ForeignKeyField(Task)
    execution_context = ForeignKeyField(ExecutionContext)


class TaskInputDataProducts(BaseModel):
    pk = AutoField()
    task = ForeignKeyField(Task)
    data_product = ForeignKeyField(DataProduct)
    

class TaskOutputDataProducts(BaseModel):
    pk = AutoField()
    task = ForeignKeyField(Task)
    data_product = ForeignKeyField(DataProduct)
