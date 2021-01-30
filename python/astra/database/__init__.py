import json
import collections
from functools import partial
from luigi.parameter import _DictParamEncoder
from sqlalchemy import MetaData, create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base, DeferredReflection
from sdssdb.connection import SQLADatabaseConnection
from sdssdb.sqlalchemy import BaseModel

AstraBase = declarative_base(cls=(DeferredReflection, BaseModel))

class AstraDatabaseConnection(SQLADatabaseConnection):
    
    dbname = "sdss5db"
    base = AstraBase

    def create_engine(self, db_connection_string=None, echo=False, pool_size=10,
                      pool_recycle=1800, expire_on_commit=True):
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

        self.engine = create_engine(
            db_connection_string, 
            echo=echo, 
            pool_size=pool_size,
            pool_recycle=pool_recycle,
            json_serializer=partial(
                json.dumps,
                cls=_DictParamEncoder
            ),
            # Unclear whether we need this deserializer or not.
            #json_deserializer=partial(
            #    json.loads, 
            #    object_pairs_hook=collections.OrderedDict
            #)
        )
        self.metadata = MetaData(bind=self.engine)
        self.Session = scoped_session(sessionmaker(bind=self.engine, autocommit=True,
                                                expire_on_commit=expire_on_commit))


database = AstraDatabaseConnection(autoconnect=True)
# Ignore what the documentation says for sdssdb.
# Create a file called ~/.config/sdssdb/sdssdb.yml and put your connection info there.

database.set_profile("astra")


session = database.Session()
