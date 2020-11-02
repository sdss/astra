
"""
Tasks for daily running.
"""

import astra
import sqlalchemy
from sqlalchemy import create_engine
from luigi.task import flatten
from astra.tasks.base import BaseTask
from astra.tasks.io import (ApStarFile, ApVisitFile)


def get_visits(mjd, full_output=False):
    """
    Yield visits that were most recently observed on the given MJD.
    
    :param mjd:
        Modified Julian Date of the observations.
    
    :param full_output: (optional)
        Return the full output from the database row (default: False). 
        If set to `False`, only the relevant ApVisit keys will be returned.
    """

    # todo: move elsewhere and refactor.
    connection_string = "postgresql://sdss_remote@operations.sdss.org/sdss5db"

    engine = create_engine(connection_string)

    with engine.begin() as connection:

        md = sqlalchemy.MetaData(schema="apogee_drp")

        table = sqlalchemy.Table("visit", md, autoload=True, autoload_with=connection)

        s = sqlalchemy.select([table]).where(table.c.mjd == mjd)
        rows = engine.execute(s).fetchall()

        keys = [column.name for column in table.columns]

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

    connection_string = "postgresql://sdss_remote@operations.sdss.org/sdss5db"

    engine = create_engine(connection_string)

    with engine.begin() as connection:

        md = sqlalchemy.MetaData(schema="apogee_drp")

        table = sqlalchemy.Table("star", md, autoload=True, autoload_with=connection)

        s = sqlalchemy.select([table]).where(table.c.mjdend == mjd)
        rows = engine.execute(s).fetchall()

        keys = [column.name for column in table.columns]
    
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

