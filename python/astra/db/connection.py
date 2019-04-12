
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base

from astra import config, log

# TODO: allow for multiple database configurations to be given in the config
#       and then have the user specify?
database_config = config.get("database", None)
if database_config is None:
    raise RuntimeError("no database configured in Astra")

# Build a database connection string.
if database_config["host"] == "localhost":
    connection_string = "postgresql+psycopg2:///{database}"
else:
    connection_string = "postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

engine = create_engine(connection_string.format(**database_config),
                       echo=True, pool_size=10, pool_recycle=1800)

# Force SQLalchemy logging to be parsed through the Astra logger.
# TODO: verbosity is not correctly followed here.
sqlalchemy_logger = logging.getLogger("sqlalchemy")
for handler in log.handlers:
    sqlalchemy_logger.addHandler(handler)

Base = declarative_base()
#Base.metadata.reflect(bind=engine)
Session = sessionmaker(bind=engine)
session = Session()
