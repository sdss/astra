import os
import datetime
import hashlib
import json
import numpy as np
from peewee import AutoField, DateTimeField, FloatField, ForeignKeyField, IntegerField
from sdssdb.connection import DatabaseConnection, PeeweeDatabaseConnection
from sdssdb.peewee import BaseModel as _BaseModel
from collections import (ChainMap, OrderedDict)
from peewee import (
    AutoField,
    DateTimeField,
    ForeignKeyField,
    FloatField,
    IntegerField,
    IntegrityError,
    TextField,
    BooleanField,
    BigIntegerField
)
from sdss_access import SDSSPath

from astra import __version__
from astra.utils import log

_version_major, _version_minor, _version_patch = tuple(map(int, __version__.split(".")))

_database_url = os.environ.get("ASTRA_DATABASE_URL", None)
if _database_url is not None:
    from playhouse.sqlite_ext import (
        SqliteExtDatabase as AstraDatabaseConnection,
        JSONField,
    )

    log.info(f"Using ASTRA_DATABASE_URL enironment variable, and assuming a SQLite database")

    # The PeeweeDatabaseConnection assumes a postgresql database under the hood. Argh!
    database = AstraDatabaseConnection(_database_url, thread_safe=True)
    schema = None
else:
    # The documentation says we should be using the PostgresqlExtDatabase if we are using a
    # BinaryJSONField, but that class is incompatible with PeeweeDatabaseConnection, and it
    # doesn't look like we need anything different from the existing PostgresqlDatabase class.
    from playhouse.postgres_ext import BinaryJSONField as JSONField
    from playhouse.postgres_ext import ArrayField
    '''
    _database_config = config.get("astra_database", {})

    class AstraDatabaseConnection(PeeweeDatabaseConnection):
        dbname = _database_config.get("dbname", None)

    database = AstraDatabaseConnection(autoconnect=True)
    schema = _database_config.get("schema", None)

    profile = _database_config.get("profile", None)
    if profile is not None:
        try:
            database.set_profile(profile)
        except AssertionError as e:
            log.exception(e)
            log.warning(
                f"""
            Database profile '{profile}' set in Astra configuration file, but there is no database
            profile called '{profile}' found in ~/.config/sdssdb/sdssdb.yml -- it should look like:

            {profile}:
                user: [USER]
                host: [HOST]
                port: 5432
                domain: [DOMAIN]
            See https://sdssdb.readthedocs.io/en/stable/intro.html#supported-profiles for more details.
            If the profile name '{profile}' is incorrect, you can change the 'database' / 'profile' key
            in ~/.astra/astra.yml
            """
            )
    '''

    class AstraDatabaseConnection(PeeweeDatabaseConnection):
        dbname = "sdss5db"
        
    database = AstraDatabaseConnection(autoconnect=True)
    database.set_profile("astra")
    schema = "astra_ipl2"


class BaseModel(_BaseModel):
    class Meta:
        database = database
        schema = schema
        use_reflection = True


class Task(BaseModel):
    
    """Unique task counter."""

    id = AutoField()
    created = DateTimeField(default=datetime.datetime.now)



class TaskRegistry(BaseModel):

    """A task registry so we know where database outputs were populated from.""" 

    id = AutoField()
    table_name = TextField()
    task_function = TextField()
    

class Source(BaseModel):

    """An astronomical source in the SDSS catalog database."""

    catalogid = BigIntegerField(primary_key=True)
    
    catalogid_v0p5 = BigIntegerField(null=True)
    catalogid_v1 = BigIntegerField(null=True)
    
    gaia_dr3_source_id = BigIntegerField(null=True)
    tic_v8_id = BigIntegerField(null=True)
    twomass_psc_designation = TextField(null=True)
    
    ra = FloatField(null=True)
    dec = FloatField(null=True)

    @property
    def outputs(self):
        # Remember: output tables must be imported for them to be within the dependency chain
        for expr, column in self.dependencies():
            if SDSSOutput in column.model.__mro__[1:]:
                yield from column.model.select().where(expr)

    

class DataProductKeywordsField(JSONField):
    def adapt(self, kwargs):
        # See https://github.com/sdss/astra/issues/8
        coerced = {}
        coerce_types = {
            # apVisit
            "mjd": int,
            # Some APOGEE paths were periodically screwed up in the database.
            "field": lambda _: str(_).strip(),
            "fiber": int,
            "apred": str,
            "healpix": int,
            # specFull
            "fieldid": int,
        }
        for key, value in kwargs.items():
            key = key.strip().lower()
            if key in coerce_types:
                value = coerce_types[key](value)
            coerced[key] = value
        return coerced

_template_dpkwf = DataProductKeywordsField()


class DataProduct(BaseModel):

    id = AutoField()
    release = TextField()
    filetype = TextField()
    kwargs = DataProductKeywordsField()
    kwargs_hash = TextField()

    metadata = JSONField(null=True)

    created = DateTimeField(default=datetime.datetime.now)
    updated = DateTimeField(default=datetime.datetime.now)

    source = ForeignKeyField(Source, null=True, backref="data_products")

    class Meta:
        indexes = (
            # Always remember to put the comma at the end.
            (("release", "filetype", "kwargs_hash"), True),
        )

    def __init__(self, *args, **kwargs):
        # Adapt keywords
        adapted, hashed = self.adapt_and_hash_kwargs(kwargs.get("kwargs", {}))
        kwargs["kwargs"] = adapted
        kwargs.setdefault("kwargs_hash", hashed)
        super(DataProduct, self).__init__(*args, **kwargs)

    def __repr__(self):
        return f"<{self.__class__.__name__} ({self.filetype}): id={self.id}>"

    @property
    def outputs(self):
        # Remember: output tables must be imported for them to be within the dependency chain
        for expr, column in self.dependencies():
            if SDSSOutput in column.model.__mro__[1:]:
                yield from column.model.select().where(expr)


    @classmethod
    def adapt_and_hash_kwargs(cls, kwargs):
        adapted = _template_dpkwf.adapt(kwargs)
        hashed = hashlib.md5(json.dumps(adapted).encode("utf-8")).hexdigest()
        return (adapted, hashed)

    @classmethod
    def get_or_create(cls, **kwargs):
        defaults = kwargs.pop("defaults", {})
        query = cls.select()

        for field, value in kwargs.items():
            # Just search by kwargs hash
            if field == "kwargs":
                # instead, seach by kwargs hash
                adapted, hashed = cls.adapt_and_hash_kwargs(value)
                query = query.where(getattr(cls, "kwargs_hash") == hashed)
            else:
                query = query.where(getattr(cls, field) == value)

        try:
            return query.get(), False
        except cls.DoesNotExist:
            try:
                if defaults:
                    kwargs.update(defaults)
                with cls._meta.database.atomic():
                    return cls.create(**kwargs), True
            except IntegrityError as exc:
                try:
                    return query.get(), False
                except cls.DoesNotExist:
                    raise exc

    @property
    def path(self):
        return SDSSPath(self.release).full(self.filetype, **self.kwargs)


class BaseTaskOutput(BaseModel):
    
    """Base table for all task outputs."""

    task = ForeignKeyField(
        Task, 
        default=Task.create,
        on_delete="CASCADE", 
        primary_key=True
    )

    time_elapsed = FloatField(null=True)
    time_bundle = FloatField(null=True)
    completed = DateTimeField(default=datetime.datetime.now)

    version_major = IntegerField(default=_version_major)
    version_minor = IntegerField(default=_version_minor)
    version_patch = IntegerField(default=_version_patch)

    data_product = ForeignKeyField(DataProduct, null=True)
    source = ForeignKeyField(Source, null=True)


# If you ever wanted to separate 'astra' from SDSS, this is the line.


def _get_meta_key(meta, key, default_value=None):
    for k in (key, key.lower(), key.upper()):
        if k in meta:
            value = meta[k]
            break
    else:
        value = default_value
    if isinstance(value, (np.ndarray, )) and value.size == 1:
        value = value.flatten()[0]
    return value



def _get_meta_keys(meta, keys, default_value=None):
    for key in keys:
        v = _get_meta_key(meta, key, default_value=default_value)
        if v is not None:
            return v
    return v



def _infer_instrument(data_product, spectrum):
    if data_product is None and spectrum is None:
        return None
    
    if "apred" in data_product.kwargs:
        return "APOGEE"
    if "run2d" in data_product.kwargs:
        return "BOSS"
    return None
    

def _get_sdss_metadata(data_product=None, spectrum=None, **kwargs):
    if spectrum is None and data_product is None:
        return {}

    telescope = _get_meta_key(spectrum.meta, "TELESCOPE", data_product.kwargs.get("telescope", None))
    if telescope is None:
        telescope = _get_meta_key(spectrum.meta, "OBSRVTRY", None)
        if isinstance(telescope, str):
            telescope = telescope.lower() + "25m"

    meta = dict(
        snr=_get_meta_key(spectrum.meta, "SNR", None),
        obj=_get_meta_key(spectrum.meta, "OBJID", None),
        mjd=_get_meta_key(spectrum.meta, "MJD", data_product.kwargs.get("mjd", None)),
        telescope=telescope,
        instrument=_get_meta_key(spectrum.meta, "INSTRMNT", _infer_instrument(data_product, spectrum)),      
        plate=_get_meta_key(spectrum.meta, "PLATE", data_product.kwargs.get("plate", None)),
        field=_get_meta_keys(spectrum.meta, ("FIELD", "FIELDID"), data_product.kwargs.get("field", None)),
        fiber=_get_meta_keys(spectrum.meta, ("FIBER", "FIBERID"), data_product.kwargs.get("fiber", None)),
        apvisit_pk=_get_meta_keys(spectrum.meta, ("VISIT_PK", ), None),
        apstar_pk=_get_meta_keys(spectrum.meta, ("STAR_PK", ), None),
    )    
    return meta


class SDSSOutput(BaseTaskOutput):
    
    # Define the metadata we want recorded for every SDSS analysis task.
    snr = FloatField(null=True)
    obj = TextField(null=True)
    mjd = FloatField(null=True)
    plate = TextField(null=True)
    field = TextField(null=True)
    telescope = TextField(null=True)
    instrument = TextField(null=True)
    fiber = IntegerField(null=True)
    apvisit_pk = IntegerField(null=True) # set foreign relational key to apvisit table
    apstar_pk = IntegerField(null=True) # set foreign relational key to apstar table

    output_data_product = ForeignKeyField(DataProduct, null=True)

    def __init__(self, data_product, spectrum=None, **kwargs):

        try:
            kwds = _get_sdss_metadata(data_product, spectrum, **kwargs)
        except:
            if spectrum is not None:
                log.exception(f"Unable to get metadata for spectrum in data product {data_product} and {spectrum}")
            kwds = kwargs
        else:
            # Inject metadata
            kwds.update(kwargs)
        try:
            kwds["source"] = data_product.source
        except:
            None
        
        super(SDSSOutput, self).__init__(data_product=data_product, **kwds)
        return None


def create_tables(drop_existing_tables=False, reuse_if_open=True):
    """
    Create all tables for the Astra database.

    :param drop_existing_tables: [optional]
        Drop existing tables from the database (default: false).

    :param reuse_if_open: [optional]
        Re-use existing database connection if one is open (default: true).

    :param insert_status_rows: [optional]
        Insert rows describing the Status of each task (default: true)
    """

    log.info(f"Connecting to database to create tables.")
    database.connect(reuse_if_open=reuse_if_open)
    ignore = (BaseTaskOutput, SDSSOutput, )

    models = [model for model in BaseModel.__subclasses__() if model not in ignore]
    log.info(
        f"Tables ({len(models)}): {', '.join([model.__name__ for model in models])}"
    )
    if drop_existing_tables:
        log.info(f"Dropping existing tables..")
        database.drop_tables(models)

    database.create_tables(models)

    log.info(f"Done.")
    return None

