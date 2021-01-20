
"""
Tasks for daily running.
"""

import astra
import sqlalchemy
from luigi import Config
from sqlalchemy import create_engine
from luigi.task import flatten
from astra.tasks import BaseTask
from astra.tasks.io.sdss5 import (ApStarFile, ApVisitFile)


class DailyMixin(Config):
    operations_user = astra.Parameter(
        default="sdss",
        config_path=dict(section="Daily", name="operations_user")
    )

# todo: move elsewhere and refactor.
connection_string = f"postgresql://{DailyMixin().operations_user}@operations.sdss.org/sdss5db"

engine = create_engine(connection_string)

connection = engine.connect()
md = sqlalchemy.MetaData(schema="apogee_drp")
visit_table = sqlalchemy.Table("visit", md, autoload=True, autoload_with=connection)
star_table = sqlalchemy.Table("star", md, autoload=True, autoload_with=connection)



def get_bhm_mwm_visits(mjd=None, plate=None, run2d=None):
    """
    Yield visits that were observed by BHM, but are of interest to MWM.

    :param mjd: (optional)
        Modified Julian Date of the observations (default: None).

    :param plate: (optional)
        The plate of the observations (default: None).

    :param run2d: (optional)
        The BHM reduction pipeline version (default: None).
    """

    md = sqlalchemy.MetaData(schema="catalogdb")
    table = sqlalchemy.Table("sdssv_boss_spall", md, autoload=True, autoload_with=connection)

    columns = [
        table.c.catalogid,
        table.c.fiberid,
        table.c.mjd,
        table.c.run2d,
        table.c.plate
    ]

    s = sqlalchemy.select(columns).where(table.c.firstcarton.like("mwm_%"))
    if mjd is not None:
        s = s.where(table.c.mjd == mjd)
    if run2d is not None:
        s = s.where(table.c.run2d == run2d)
    if plate is not None:
        s = s.where(table.c.plate == plate)

    rows = engine.execute(s).fetchall()
    column_names = [column.name for column in columns]
    
    for row in rows:
        yield dict(zip(column_names, row))



def get_mjds_of_unclassified_apogee_visits():
    """
    Return Modified Julian Dates (MJD) where ApVisit objects were reduced, but no 
    classifications exist for that MJD. This does not do a *complete* (all) check 
    against which MJDs have classifications. It only checks to see if there are 
    *any* MJDs for which there are APOGEE visits, but there are no classifications.
    """

    # Get distinct MJD end values from the apogee_drp.star table.
    s_star = sqlalchemy.select([star_table.c.mjdend]).distinct()

    # Get distinct MJD values from the astra.classify_apvisit table.
    # TODO: Get the table name from the actual Classify task?
    astra_metadata = sqlalchemy.MetaData(schema="astra")
    classify_apvisit = sqlalchemy.Table(
        "classify_apvisit", 
        astra_metadata, 
        autoload=True, 
        autoload_with=connection
    )

    s_classify = sqlalchemy.select([classify_apvisit.c.mjd]).distinct()

    expected_mjds = flatten(engine.execute(s_star).fetchall())
    classified_mjds = flatten(engine.execute(s_classify).fetchall())

    return tuple(sorted(set(expected_mjds).difference(classified_mjds)))

    


def get_visits(mjd, full_output=False):
    """
    Yield visits that were most recently observed on the given MJD.
    
    :param mjd:
        Modified Julian Date of the observations.
    
    :param full_output: (optional)
        Return the full output from the database row (default: False). 
        If set to `False`, only the relevant ApVisit keys will be returned.
    """

    #with :
    s = sqlalchemy.select([visit_table]).where(visit_table.c.mjd == mjd)
    rows = engine.execute(s).fetchall()

    keys = [column.name for column in visit_table.columns]

    if full_output:
        # TODO: Consider translating so output behaviour is consistent.
        for row in rows:
            yield dict(zip(keys, row))

    only_keys = ("fiber", "plate", "mjd", "telescope", "field", "apred")
    translate_keys = {
        "fiber": "fiberid",
        "apred": "apred_vers"
    }
    for row in rows:
        yield dict(zip(only_keys, [row[translate_keys.get(k, k)] for k in only_keys]))


def get_stars(mjd, full_output=False):
    """
    Yield stars that were most recently observed on the given MJD.

    :param mjd:
        The Modified Julian Date of the most recent observations.

    :param full_output: (optional)
        Return the full output from the database row (default: False). 
        If set to `False`, only the relevant ApStar keys will be returned.
    """

    #with engine.begin() as connection:

    s = sqlalchemy.select([star_table]).where(
            (star_table.c.mjdend == mjd) \
        &   (star_table.c.ngoodrvs > 0)
    )
    rows = engine.execute(s).fetchall()

    keys = [column.name for column in star_table.columns]

    if full_output:
        for row in rows:
            yield dict(zip(keys, row))
    
    only_keys = ("apred", "healpix", "telescope", "obj")
    translate_keys = {
        "apred": "apred_vers",
        "obj": "apogee_id"
    }
    for row in rows:
        d = dict(zip(only_keys, [row[translate_keys.get(k, k)] for k in only_keys]))
        d.update(apstar="stars")
        yield d



class GetDailyMixin(BaseTask):

    """ A mixin class for daily data product retrival tasks. """

    mjd = astra.IntParameter()

    release = astra.Parameter(default="sdss5")
    public = astra.BoolParameter(default=False)
    use_remote = astra.BoolParameter(default=True)

    def complete(self):
        return all(r.complete() for r in flatten(self.requires()))



class GetDailyApVisitFiles(GetDailyMixin):

    """
    A wrapper task that requires all daily APOGEE visits, given some Modified Julian Date.

    :param mjd:
        Modified Julian Date of observations.

    :param release: (optional)
        The name of the data release (default: sdss5).
    
    :param public: (optional)
        A boolean flag indicating whether the data are public or not (default: False).
    
    :param use_remote: (optional)
        Download the file from the remote server if it does not exist locally (default: True).    
    """

    def requires(self):
        common_kwds = self.get_common_param_kwargs(ApVisitFile)
        for kwds in get_visits(self.mjd, full_output=False):
            yield ApVisitFile(**{ **common_kwds, **kwds })


class GetDailyApStarFiles(GetDailyMixin):

    """
    A wrapper task that requires all daily APOGEE star files, given some Modified Julian Date.

    :param mjd:
        Modified Julian Date of observations.

    :param release: (optional)
        The name of the data release (default: sdss5).
    
    :param public: (optional)
        A boolean flag indicating whether the data are public or not (default: False).
    
    :param use_remote: (optional)
        Download the file from the remote server if it does not exist locally (default: True).    
    """

    def requires(self):
        common_kwds = self.get_common_param_kwargs(ApStarFile)
        for kwds in get_stars(self.mjd, full_output=False):
            # TODO: We will need some logic here to figure out if we have previously run
            #       this task and need to re-download it because the MJD has changed...
            #       (This will not be a problem when running at Utah)
            yield ApStarFile(**{ **common_kwds, **kwds })


class GetDailyReducedDataProducts(GetDailyMixin):

    """
    A wrapper task that yields all new SDSS-V/MWM reduced data products, given some Modified Julian Date.

    :param mjd:
        Modified Julian Date of observations.

    :param release: (optional)
        The name of the data release (default: sdss5).
    
    :param public: (optional)
        A boolean flag indicating whether the data are public or not (default: False).
    
    :param use_remote: (optional)
        Download the file from the remote server if it does not exist locally (default: True).    
    """

    def requires(self):
        yield GetDailyApStarFiles(**self.param_kwargs)
        yield GetDailyApVisitFiles(**self.param_kwargs)
        # TODO: Get BOSS spectra of stars.
        


# TODO: This should go somewhere else, probably.
def get_visits_given_star(obj, apred):
    """
    Get the keywords of individual visits (ApVisit files) that went into a star (ApStar) file.

    :param obj:
        The name of the object (e.g., 2M00000+000000).
    
    :param apred:
        The version of the APOGEE reduction pipeline used (e.g., daily).
    """
    
    columns = [
        visit_table.c.fiberid.label("fiber"),
        visit_table.c.plate,
        visit_table.c.mjd,
        visit_table.c.telescope,
        visit_table.c.field,
        visit_table.c.apred_vers.label("apred"),
    ]

    s = sqlalchemy.select(columns).where(
            (visit_table.c.apogee_id == obj) & (visit_table.c.apred_vers == apred)
        )

    rows = engine.execute(s).fetchall()
    keys = [column.name for column in columns]

    for row in rows:
        yield dict(zip(keys, row))
    



"""
if __name__ == "__main__":

    mjd = 59146
    task = GetDailyReducedDataProducts(mjd=59146)
    
    astra.build(
        [task],
        local_scheduler=True
    )

    # TODO: Ask Nidever to change apogee_drp.visit schema to be consistent with datamodel. Specifically:
    #       fiberid -> fiber
    #       apred_vers -> apred
    #       apogee_id -> obj
    #       remove apstar?

"""
