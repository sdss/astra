
__all__ = ["BaseModel", "database"]

import os
import re
import numpy as np

from peewee import (
    Field,
    Model,
    PostgresqlDatabase,
    TextField,
    FloatField,
    BooleanField,
    IntegerField,
    AutoField,
    BigIntegerField,
    ForeignKeyField,
    DateTimeField,
    JOIN
)
from inspect import getsource
from playhouse.sqlite_ext import SqliteExtDatabase
from sdsstools.configuration import get_config

from astra.fields import BitField
from astra.utils import log, get_config_paths, expand_path

BLANK_CARD = (" ", " ", None)
FILLER_CARD = (FILLER_CARD_KEY, *_) = ("TTYPE0", "Water cuggle", None)



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

                    from peewee import OperationalError, __exception_wrapper__


                    class RetryOperationalError(object):

                        def execute_sql(self, sql, params=None, commit=True):
                            try:
                                cursor = (
                                    super(RetryOperationalError, self)
                                    .execute_sql(sql, params, commit)
                                )
                            except OperationalError:
                                if not self.is_closed():
                                    self.close()
                                with __exception_wrapper__:
                                    cursor = self.cursor()
                                    cursor.execute(sql, params or ())
                                    if commit and not self.in_transaction():
                                        self.commit()
                            return cursor

                    class _PostgresqlDatabase(RetryOperationalError, PostgresqlDatabase):
                        pass

                    database = _PostgresqlDatabase(
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

# Doing this here to avoid circular import for config.. TODO
from  pathlib import Path
database, schema = get_database_and_schema(
    get_config(
        "astra",
        config_file=f"{Path(__file__).parent.resolve()}/../etc/astra.yml"
    )   
)
if os.environ.get("FOO", False):
    schema = "astra_043"
    print("SETTING SCHEMA")

class BaseModel(Model):
    
    class Meta:
        database = database
        schema = schema
        legacy_table_names = False


    @classmethod
    @property
    def category_headers(cls):
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


    @classmethod
    @property
    def category_comments(cls):
        pattern = '\s{4}([\w|\d|_]+)\s*=.+\n\s{4}#<\s*(.+)'
        source_code = getsource(cls)
        comments = []
        for field_name, comment in re.findall(pattern, source_code):
            if hasattr(cls, field_name) and isinstance(getattr(cls, field_name), Field):
                comments.append((comment, field_name))
        return tuple(comments)

    @property
    def absolute_path(self):
        """Absolute path of `self.path`."""
        return expand_path(self.path)



def add_category_comments(hdu, models, original_names, upper):
    category_comments_added = []
    list_original_names = list(original_names.values())
    for model in models:
        for comment, field_name in model.category_comments:
            if field_name in category_comments_added:
                continue
            try:
                index = 1 + list_original_names.index(field_name)
            except:
                continue
            key = f"TFORM{index}"
            hdu.header.insert(key, ("COMMENT", comment), after=True)
            category_comments_added.append(field_name)
    return None

def add_category_headers(hdu, models, original_names, upper):
    category_headers_added = []
    list_original_names = list(original_names.values())
    for model in models:
        for header, field_name in model.category_headers:
            if field_name in category_headers_added:
                continue
            try:
                index = 1 + list_original_names.index(field_name)
            except:
                continue
            key = f"TTYPE{index}"
            hdu.header.insert(key, BLANK_CARD)
            hdu.header.insert(key, (" ", header.upper() if upper else header))
            hdu.header.insert(key, BLANK_CARD)
            category_headers_added.append(field_name)
    
    return None


def fits_column_kwargs(field, values, upper, warn_comment_length=47, warn_total_length=65):
    mappings = {
        # Require at least one character for text fields
        TextField: lambda v: dict(format="A{}".format(max(1, max(len(_) for _ in v)) if len(v) > 0 else 1)),
        BooleanField: lambda v: dict(format="L"),
        IntegerField: lambda v: dict(format="J"),
        FloatField: lambda v: dict(format="E"), # single precision
        AutoField: lambda v: dict(format="K"),
        BigIntegerField: lambda v: dict(format="K"),
        # We are assuming here that all foreign key fields are big integers
        ForeignKeyField: lambda v: dict(format="K"),
        BitField: lambda v: dict(format="J"), # integer
        DateTimeField: lambda v: dict(format="A26")
    }
    callable = mappings[type(field)]

    if isinstance(field, DateTimeField):
        array = []
        for value in values:
            try:
                array.append(value.isoformat())
            except:
                array.append(value)
    else:
        array = values

    kwds = dict(
        name=field.name.upper() if upper else field.name,
        array=array,
        unit=None,
    )
    kwds.update(callable(values))
    return kwds


def warn_on_long_name_or_comment(field, warn_comment_length=47, warn_total_length=65):
    total = len(field.name)
    if field.help_text is not None:
        if len(field.help_text) > warn_comment_length:
            log.warning(f"Field {field} help text is too long for FITS header ({len(field.help_text)} > {warn_comment_length}).")
        total += len(field.help_text)
    if total > warn_total_length:
        log.warning(f"Field {field} name and help text are too long for FITS header ({total} > {warn_total_length}).")
    return None


def get_fill_value(field, given_fill_values):
    try:
        return given_fill_values[field.name]
    except:
        try:
            if field.default is not None:
                return field.default
        except:
            None
        finally:
            default_fill_values = {
                TextField: "",
                BooleanField: False,
                IntegerField: -1,
                AutoField: -1,
                BigIntegerField: -1,
                FloatField: np.nan,
                ForeignKeyField: -1,
                DateTimeField: "",
                BitField: 0            
            }
            return default_fill_values[type(field)]
                
