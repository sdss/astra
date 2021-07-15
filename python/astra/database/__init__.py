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

"""
Reader beware:

- If you want to use Astra with only a single worker, then there's no problem.

- If you want to use multiple workers then you have a problem. The AstraDatabaseConnection
  cannot be passed to child processes, otherwise everything fucks up. Two ameliorate this
  you have two options (not really, you only have Option #2):

  1. Set use_pooling = False to disable pooling on the database connection. 
  
     That means every time a transaction is to occur, it will create a connection to the
     database and close it when it's done. This causes some overhead, but will be fine
     if you only have a couple of workers that are not interacting with the database much.

     If you have a lot of workers, or few workers who are accessing the database a lot,
     then this can cause issues on the PostgreSQL server: too many incoming connections
     at once will make the server just stop accepting new connections. Then you have
     issues with tasks failing because they cannot transmit their results to the
     database.

  2. Set use_pooling = True, but require that every child process runs `init_process(database)`
     *ONCE* -- and only once -- the moment it is created. This will make the database dispose
     of itself in the child process (`database.engine.dispose()`), and force it to reconnect.

     That's good, but when should Astra do that?! We cannot do it when a task is run, because
     database connections are needed to check if a task is complete. We cannot even do it
     during the `complete()` method -- which is run first by luigi when a child process is
     created -- because that `complete()` method will be run many times. We have to run this
     step the moment that the child process is created, and never think of it again.

     To do this we need to:
    
        -> Set "core.parallel_scheduling = True" in utah.cfg like:

            [core]
            parallel_scheduling=True
        
        -> Set use_pooling = True

        -> Change luigi/worker.py around line 747 to give the `initializer` and `initargs`
           keyword arguments for the `multiprocessing.Pool()` when a task is added (`add()`)
           to a worker. Here is what it looks like:

            # Top of the file:
            from astra.database import init_process, database

            # Around line 747:

            pool = multiprocessing.Pool(
                initializer=init_process, initargs=(database, ),
                processes=processes if processes > 0 else None
            )

    This seems to work. Until luigi allows for custom kwargs to be passed to the multiprocessing.Pool,
    we will need to use a butchered version of luigi that has this functionality.

"""


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

except AssertionError:
    from astra.utils import log
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
