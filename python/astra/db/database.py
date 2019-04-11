
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy_utils import database_exists, create_database
# TODO: pgpasslib not on conda-forge. Import only as needed? Remove?
#from pgpasslib import getpass

from astra import config, log

# TODO: allow for multiple database configurations to be given in the config
#       and then have the user specify?
database_config = config.get("database", None)
if database_config is None:
    raise RuntimeError("no database configured in Astra")

# Check for password
#if "password" not in database_config:
#    database_config["password"] = getpass(**database_config)

# Build a database connection string.
if database_config["host"] == "localhost":
    connection_string = "postgresql+psycopg2:///{database}"
else:
    connection_string = "postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

engine = create_engine(connection_string.format(**database_config),
                       echo=True, pool_size=10, pool_recycle=1800)
if not database_exists(engine.url):
    log.info(f"Creating database {engine.url}")
    create_database(engine.url)


metadata = MetaData()
metadata.bind = engine
base = declarative_base(bind=engine)
session = sessionmaker(bind=engine, autocommit=True)

