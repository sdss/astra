from astra import log
from astra.migrations.sdss5db.utils import get_profile

try:
    profile = get_profile()

    from sdssdb.peewee import BaseModel as _BaseModel, ReflectMeta
    from sdssdb.connection import PeeweeDatabaseConnection

    database = PeeweeDatabaseConnection(
        "sdss5db",
        #autorollback=True
    )
    database.set_profile(profile)

    class BaseModel(_BaseModel, metaclass=ReflectMeta):
        class Meta:
            primary_key = False
            use_reflection = False
            database = database

except:
    log.exception(f"Exception when trying to create sdss5db reflected base model. SDSS-V migrations won't work!")
    
    # TODO: use a database proxy
    BaseModel = object