import json
import os
from functools import partial
from luigi.parameter import _DictParamEncoder
from sqlalchemy import MetaData, create_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base, DeferredReflection
from sdssdb.connection import SQLADatabaseConnection
from sdssdb.sqlalchemy import BaseModel

AstraBase = declarative_base(cls=(DeferredReflection, BaseModel))


class AstraDatabaseConnection(SQLADatabaseConnection):
    
    dbname = "sdss5db"
    base = AstraBase

    def create_engine(
            self, 
            db_connection_string=None, 
            echo=False, 
            use_pooling=True,
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
            db_connection_string = self._make_connection_string(dbname,
                                                                **self.connection_params)

        if use_pooling:
            # Some defaults for pooling:
            kwds = dict(pool_size=5, pool_recycle=1800)
            kwds.update(kwargs)
        else:
            # No pooling. Necessary when using multiple workers.
            # See https://docs.sqlalchemy.org/en/14/core/pooling.html#pooling-multiprocessing
            kwds = dict(poolclass=NullPool)

        self.engine = create_engine(
            db_connection_string, 
            echo=echo, 
            pool_pre_ping=pool_pre_ping,
            json_serializer=partial(
                json.dumps,
                cls=_DictParamEncoder
            ),
            **kwds
        )
        self.metadata = MetaData(bind=self.engine)
        self.Session = scoped_session(sessionmaker(bind=self.engine, autocommit=True,
                                                expire_on_commit=expire_on_commit))


database = AstraDatabaseConnection(autoconnect=True)
# Ignore what the documentation says for sdssdb.
# Create a file called ~/.config/sdssdb/sdssdb.yml and put your connection info there.

try:
    database.set_profile("astra")

except AssertionError as e:
    from astra.utils import log
    log.exception(e)
    log.warning(""" No database profile named 'astra' found in ~/.config/sdssdb/sdssdb.yml -- it should look like:

        astra:
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
