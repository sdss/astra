import json
import os
import numpy

from functools import partial
from sqlalchemy import MetaData, create_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base, DeferredReflection
from sdssdb.connection import SQLADatabaseConnection
from sdssdb.sqlalchemy import BaseModel
from psycopg2.extensions import register_adapter, AsIs

SDSSBase = declarative_base(cls=(DeferredReflection, BaseModel))


class SDSSDatabaseConnection(SQLADatabaseConnection):
    
    dbname = "sdss5db"
    base = SDSSBase

    def create_engine(
            self, 
            db_connection_string=None, 
            echo=False, 
            pool_size=5,
            pool_recycle=1800,
            expire_on_commit=True,
            pool_pre_ping=True,
            **kwargs
        ):
        '''
        Create a new database engine

        Resets and creates a new sqlalchemy database engine.  Also creates and binds
        engine metadata and a new scoped session.
        '''

        self.reset_engine()

        if not db_connection_string:
            dbname = self.dbname or self.DATABASE_NAME
            db_connection_string = self._make_connection_string(
                dbname,
                **self.connection_params
            )

        self.engine = create_engine(
            db_connection_string, 
            echo=echo, 
            pool_size=pool_size,
            pool_recycle=pool_recycle,
            pool_pre_ping=pool_pre_ping,
        )
        self.metadata = MetaData(bind=self.engine)
        self.Session = scoped_session(
            sessionmaker(
                bind=self.engine, 
                autocommit=True,
                expire_on_commit=expire_on_commit
            )
        )


database = SDSSDatabaseConnection(autoconnect=True)
# Ignore what the documentation says for sdssdb.
# Create a file called ~/.config/sdssdb/sdssdb.yml and put your connection info there.

try:
    database.set_profile("astra")

except AssertionError as e:
    from astra.utils import log
    log.exception(e)
    log.warning(""" No database profile named 'SDSS' found in ~/.config/sdssdb/sdssdb.yml -- it should look like:

        SDSS:
          user: [SDSSDB_USERNAME]
          host: [SDSSDB_HOSTNAME]
          port: 5432
          domain: [SDSSDB_DOMAIN]

        See https://sdssdb.readthedocs.io/en/stable/intro.html#supported-profiles for more details. 
        """
    )
    session = None

else:
    try:
        session = database.Session()
    except:
        print(f"Cannot load database session")


def init_process(database):
    """
    Dispose of the existing database engine.

    This is a necessary step as soon as a child process is created.
    """
    print(f"Disposing engine {database.engine} because we are in a child process (PID = {os.getpid()}).")
    database.engine.dispose()


register_adapter(numpy.float64, AsIs)
register_adapter(numpy.float32, AsIs)
