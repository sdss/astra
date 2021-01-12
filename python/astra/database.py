from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import Session
from astra.utils import get_default
from astra.tasks.base import BaseTask
from sqlalchemy.ext.automap import automap_base

# TODO: This caused an error when running on the GPU node with The Payne.
#connection_string = get_default(BaseTask, "connection_string")
connection_string = "postgresql://sdss@operations.sdss.org/sdss5db"
engine = create_engine(connection_string)
default_schemas = ("astra", "apogee_drp")


def session_maker(schema_names=None):
    f"""
    Return a database session and schema collections.

    :param schema_names: [optional]
        A tuple of schema names to auto-map the session to.
        If `None` is provided then the default schema are:
        {default_schemas}
    
    :returns:
        A two-length tuple containing the database session,
        and the schema collections.
    """

    class Schemas(object):
        pass
    
    schema = Schemas()
    
    if isinstance(schema_names, (str, bytes)):
        schema_names = tuple(schema_names)
    
    for schema_name in schema_names:
        metadata = MetaData(schema=schema_name)

        Base = automap_base(bind=engine, metadata=metadata)
        Base.prepare(engine, reflect=True)

        setattr(schema, schema_name, Base.classes)

    session = Session(engine)

    return (session, schema)


def automap_table(schema_name, table_name):
    """
    Auto-map a table from the database, given a schema name and a table name.

    :param schema_name:
        The name of the schema that the table belongs to in the database.
    
    :param table_name:
        The name of the table.
    
    :returns:
        An auto-mapped class of the table.
    """
    connection = engine.connect()
    metadata = MetaData(schema=schema_name)

    return Table(
        table_name,
        metadata, 
        autoload=True, 
        autoload_with=connection
    )



session, schema = session_maker(default_schemas)
