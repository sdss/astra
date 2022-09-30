from sdssdb.peewee import BaseModel, ReflectMeta
from sdssdb.connection import PeeweeDatabaseConnection
from astra import config, log

_database_config = config.get("sdss5_database", {})
database = PeeweeDatabaseConnection(_database_config["dbname"])

# TODO: DRY this up with astradb.py
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


class ReflectBaseModel(BaseModel, metaclass=ReflectMeta):
    class Meta:
        primary_key = False
        use_reflection = False
        database = database
