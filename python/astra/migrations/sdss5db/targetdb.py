from sdssdb.peewee.sdss5db.targetdb import database
from astra.migrations.sdss5db.utils import get_profile

try:
    database.set_profile(get_profile())
except (KeyError, TypeError):
    None

from sdssdb.peewee.sdss5db.targetdb import *