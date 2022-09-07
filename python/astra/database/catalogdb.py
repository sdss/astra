from sdssdb.peewee.sdss5db.catalogdb import database
from astra import config

try:
    database.set_profile(config["sdss5_database"]["profile"])
except (KeyError, TypeError):
    None

from sdssdb.peewee.sdss5db.catalogdb import *
