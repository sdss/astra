
from sqlalchemy import func

from astra.utils import log

from astra.database import (apogee_drpdb, catalogdb, session)
from astra.operators.base import AstraOperator
from astra.operators.utils import (parse_as_mjd, infer_release)

class ApVisitOperator(AstraOperator):
    """
    A base operator for working with SDSS ApVisit data products. 
    
    This operator will generate task instances based on ApVisit data products it finds that were
    completed in the operator execution period.

    :param release: [optional]
        The relevant SDSS data release. If `None` is given then this will be inferred based on
        the execution date.
    :type release: str
    :param _limit: [optional]
        Limit the database query to only return `_limit` ApVisit objects.
    :type _limit: int
    :param _query_filter_by_kwargs: [optional]
        A dictionary of keyword arguments to give to the sqlalchemy query 
        (as `q.filter_by(**_query_filter_by_kwargs)`).
    :type _query_filter_by_kwargs: dict

    Additional parameters that are inherited from the AstraOperator class:

    :param spectrum_callback: [optional]
        The name of a function to execute on spectra after they are loaded 
        (e.g., 'astra.tools.continuum.mean').
    :type spectrum_callback: str
    :param spectrum_callback_kwargs: [optional]
        A dictionary of keyword arguments that will be submitted to the `spectrum_callback`.
    :param slurm_kwargs: [optional]
        A dictionary of keyword arguments that will be submitted to the slurm queue.
    :type slurm_kwargs: dict
    :param _data_model_identifiers: [optional]
        Directly supply data model identifiers to this operator and *do not* find data model
        identifiers for the given DAG execution context. Only use this if you know what you
        are doing. Using this argument will override the default behaviour of the operators,
        and will make the operator only process the data that is given by `_data_model_identifiers`,
        regardless of the execution date. This means the operator will process the same set
        of data every time it is triggered! This argument should only be used for testing.
        An example use of this argument might be:

        >> _data_model_identifiers = [
        >>     {
        >>         "obj": "2M06343180+0546387", 
        >>         "telescope": "apo25m",
        >>         "apstar": "stars",
        >>         "healpix": "88460",
        >>         "apred": "daily",
        >>         "filetype": "ApStar", 
        >>         "release": "sdss5"
        >>     }
        >> ]

    :type _data_model_identifiers:
        An iterable that includes dictionaries that fully define a data model product, or a
        callable function that returns an iterable.
    """

    def __init__(
        self,
        *,
        release = None,
        # We want to be able to supply these arguments, but we don't want them stored as parameters
        # in the task instances, so we prefix them with '_'.
        _limit = None,
        _query_filter_by_kwargs = None,
        **kwargs,
    ) -> None:
        super(ApVisitOperator, self).__init__(**kwargs)
        self.release = release
        self._limit = _limit
        self._query_filter_by_kwargs = _query_filter_by_kwargs
    

    def query_data_model_identifiers_from_database(self, context):
        """
        Query the SDSS database for ApVisit data model identifiers.

        :param context:
            The Airflow DAG execution context.
        """ 
        release = self.release or infer_release(context)
        mjd_start = parse_as_mjd(context["prev_ds"])
        mjd_end = parse_as_mjd(context["ds"])
        log.debug(f"Parsed MJD range as between {mjd_start} and {mjd_end}")

        if release.lower() in ("dr16", ):
            yield from self.query_sdss4_dr16_data_model_identifiers_from_database(mjd_start, mjd_end)
        else:
            yield from self.query_sdss5_data_model_identifiers_from_database(mjd_start, mjd_end)


    def query_sdss4_dr16_data_model_identifiers_from_database(self, mjd_start, mjd_end):
        """
        Query the SDSS database for SDSS-IV (DR16) ApVisit data model identifiers.

        :param mjd_start:
            The starting Modified Julian Date (MJD) of observations.

        :param mjd_end:
            The ending Modified Julian Date (MJD) of observations. Observations are
            returned if they exist between (start <= time < end).        
        """ 

        release, filetype = (self.release, "apVisit")

        columns = (
            func.left(catalogdb.SDSSDR16ApogeeVisit.file, 2).label("prefix"),
            catalogdb.SDSSDR16ApogeeVisit.plate,
            catalogdb.SDSSDR16ApogeeVisit.location_id.label("field"),
            catalogdb.SDSSDR16ApogeeVisit.telescope,
            catalogdb.SDSSDR16ApogeeVisit.mjd,
            catalogdb.SDSSDR16ApogeeVisit.fiberid.label("fiber"),
            catalogdb.SDSSDR16ApogeeVisit.apred_version.label("apred")
        )
        q = session.query(*columns).distinct(*columns)
        q = q.filter(catalogdb.SDSSDR16ApogeeVisit.mjd >= mjd_start)\
             .filter(catalogdb.SDSSDR16ApogeeVisit.mjd < mjd_end)
        
        if self._query_filter_by_kwargs is not None:
            q = q.filter_by(**self._query_filter_by_kwargs)

        if self._limit is not None:
            q = q.limit(self._limit)

        log.debug(f"Found {q.count()} {release} {filetype} files between MJD {mjd_start} and {mjd_end}")

        common = dict(release=release, filetype=filetype)
        keys = [column.name for column in columns]
        for values in q.yield_per(1):
            yield { **common, **dict(zip(keys, values)) }
                

    def query_sdss5_data_model_identifiers_from_database(self, mjd_start, mjd_end):
        """
        Query the SDSS-V database for ApVisit data model identifiers.
        
        :param mjd_start:
            The starting Modified Julian Date (MJD) of observations.

        :param mjd_end:
            The ending Modified Julian Date (MJD) of observations. Observations are
            returned if they exist between (start <= time < end).
        """

        release, filetype = (self.release, "apVisit")

        columns = (
            apogee_drpdb.Visit.apogee_id.label("obj"), # TODO: Raise with Nidever
            apogee_drpdb.Visit.telescope,
            apogee_drpdb.Visit.fiberid.label("fiber"), # TODO: Raise with Nidever
            apogee_drpdb.Visit.plate,
            apogee_drpdb.Visit.field,
            apogee_drpdb.Visit.mjd,
            apogee_drpdb.Visit.apred_vers.label("apred"), # TODO: Raise with Nidever
            func.left(apogee_drpdb.Visit.file, 2).label("prefix")
        )
        q = session.query(*columns).distinct(*columns)
        q = q.filter(apogee_drpdb.Visit.mjd >= mjd_start)\
             .filter(apogee_drpdb.Visit.mjd < mjd_end)
        
        if self._query_filter_by_kwargs is not None:
            q = q.filter_by(**self._query_filter_by_kwargs)

        if self._limit is not None:
            q = q.limit(self._limit)

        log.debug(f"Found {q.count()} {release} {filetype} files between MJD {mjd_start} and {mjd_end}")

        common = dict(release=release, filetype=filetype)
        keys = [column.name for column in columns]
        for values in q.yield_per(1):
            yield { **common, **dict(zip(keys, values)) }
                

