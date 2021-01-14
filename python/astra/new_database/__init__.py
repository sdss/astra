from sqlalchemy.ext.declarative import declarative_base, DeferredReflection
from sdssdb.connection import SQLADatabaseConnection
from sdssdb.sqlalchemy import BaseModel


AstraBase = declarative_base(cls=(DeferredReflection, BaseModel))


class AstraDatabaseConnection(SQLADatabaseConnection):
    dbname = "sdss5db"
    base = AstraBase


database = AstraDatabaseConnection(autoconnect=True)
# Ignore what the documentation says for sdssdb.
# Create a file called ~/.config/sdssdb/sdssdb.yml and put your connection info there.

database.set_profile("astra")

