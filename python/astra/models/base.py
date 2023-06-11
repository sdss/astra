
__all__ = ["BaseModel", "database"]

import os
import re
from inspect import getsource
from peewee import (Field, Model, PostgresqlDatabase)
from playhouse.sqlite_ext import SqliteExtDatabase

from astra import config
from astra.utils import log, get_config_paths, expand_path

# Note that we can't use a DatabaseProxy and define it later because we also need to be
# able to dynamically set the schema, which seems to be impossible with a DatabaseProxy.

def get_database_and_schema(config):
    """
    Return a database and schema given some configuration.
    
    :param config:
        A dictionary of configuration values, read from a config file.
    """
        
    sqlite_kwargs = dict(
        thread_safe=True,
        pragmas={
            'journal_mode': 'wal',
            'cache_size': -1 * 64000,  # 64MB
            'foreign_keys': 1,
            'ignore_check_constraints': 0,
            'synchronous': 0
        }
    )

    config_placement_message = (
        "These are the places where Astra looks for a config file:\n"
    )
    for path in get_config_paths():
        config_placement_message += f"  - {path}\n"    
    config_placement_message += (
        "\n\n"
        "This is what the `database` entry could look like in the config file:\n"
        "  # For PostgreSQL (preferred)\n"
        "  database:\n"
        "    dbname: <DATABASE_NAME>\n"
        "    user: [USERNAME]       # can be optional\n"
        "    host: [HOSTNAME]       # can be optional\n"
        "    password: [PASSWORD]   # can be optional\n"
        "    port: [PORT]           # can be optional\n"
        "    schema: [SCHEMA]       # can be optional\n"            
        "\n"
        "  # For SQLite\n"
        "  database:\n"
        "    path: <PATH_TO_DATABASE>\n\n"
    )

    if os.getenv("ASTRA_DATABASE_PATH"):
        database_path = os.getenv("ASTRA_DATABASE_PATH")
        log.info(f"Setting database path to {database_path}.")
        log.info(f"You can change this using the `ASTRA_DATABASE_PATH` environment variable.")
        database = SqliteExtDatabase(database_path, **sqlite_kwargs)
        return (database, None)
        
    elif config.get("TESTING", False):
        log.warning("In TESTING mode, using in-memory SQLite database")
        database = SqliteExtDatabase(":memory:", **sqlite_kwargs)
        return (database, None)

    else:
        if "database" in config and isinstance(config["database"], dict):
            # Prefer postgresql
            if "dbname" in config["database"]:
                try:
                    keys = ("user", "host", "password", "port")
                    kwds = dict([(k, config["database"][k]) for k in keys if k in config["database"]])
                    kwds.setdefault("autorollback", True)

                    '''
                    print("TODO: ANDY COME BACK TO THIS")
                    from sdssdb.connection import PeeweeDatabaseConnection
                    class AstraDatabaseConnection(PeeweeDatabaseConnection):
                        dbname = config["database"]["dbname"]
                    
                    database = AstraDatabaseConnection(autoconnect=True)
                    database.set_profile("astra")
                    '''
                    database = PostgresqlDatabase(
                        config["database"]["dbname"], 
                        **kwds
                    )

                    schema = config["database"].get("schema", None)

                except:
                    log.exception(f"Could not create PostgresqlDatabase from config.\n{config_placement_message}")
                    raise
                else:
                    return (database, schema)

            elif "path" in config["database"]:
                try:
                    database = SqliteExtDatabase(expand_path(config["database"]["path"]), **sqlite_kwargs)
                except:
                    log.exception(f"Could not create SqliteExtDatabase from config.\n{config_placement_message}")
                    raise 
                else:
                    return (database, None)

        
        log.warning(f"No valid `database` entry found in Astra config file.\n{config_placement_message}")
        log.info(f"Defaulting to in-memory SQLite database.")
        database = SqliteExtDatabase(":memory:", **sqlite_kwargs)
    
        return (database, None)



database, schema = get_database_and_schema(config)
'''
from sdssdb.connection import PeeweeDatabaseConnection
class AstraDatabaseConnection(PeeweeDatabaseConnection):
    dbname = "sdss5db"

database = AstraDatabaseConnection(autoconnect=True)
database.set_profile("astra")

schema = "astra_temp"
'''

class BaseModel(Model):
    
    class Meta:
        database = database
        schema = schema
        legacy_table_names = False


    @classmethod
    @property
    def field_category_headers(cls):
        """
        Return a tuple of category headers for the data model fields based on the source code.
        Category headers are defined in the source code like this:

        ```python
        #> Category header
        teff = FloatField(...)

        #> New category header
        """

        pattern = '\s{4}#>\s*(.+)\n\s{4}([\w|\d|_]+)\s*='
        source_code = getsource(cls)
        category_headers = []
        for header, field_name in re.findall(pattern, source_code):
            if hasattr(cls, field_name) and isinstance(getattr(cls, field_name), Field):
                category_headers.append((header, field_name))
            else:
                log.warning(
                    f"Found category header '{header}', starting above '{field_name}' in {cls}, "
                    f"but {cls}.{field_name} is not an attribute of type `peewee.Field`."
                )
        return tuple(category_headers)