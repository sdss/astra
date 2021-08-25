
from astra.utils import log

from astra.database import (catalogdb, session)
from astra.operators.base import AstraOperator
from astra.operators.utils import parse_as_mjd


class BossSpecOperator(AstraOperator):
    """
    A base operator for working with SDSS BOSS spectrum data products. 
    
    This operator will generate task instances based on BOSS spec data products it finds that were
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
        super(BossSpecOperator, self).__init__(**kwargs)
        self.release = release
        self._limit = _limit
        self._query_filter_by_kwargs = _query_filter_by_kwargs
    

    def query_data_model_identifiers_from_database(self, context):
        """
        Query the SDSS database for BOSS spectrum data model identifiers.

        :param context:
            The Airflow DAG execution context.
        """ 

        release, filetype = (self.release, "spec")
        
        mjd_start = parse_as_mjd(context["prev_ds"])
        mjd_end = parse_as_mjd(context["ds"])

        columns = (
            catalogdb.SDSSVBossSpall.catalogid,
            catalogdb.SDSSVBossSpall.run2d,
            catalogdb.SDSSVBossSpall.plate,
            catalogdb.SDSSVBossSpall.mjd,
            catalogdb.SDSSVBossSpall.fiberid
        )
        q = session.query(*columns).distinct(*columns)
        q = q.filter(catalogdb.SDSSVBossSpall.mjd >= mjd_start)\
             .filter(catalogdb.SDSSVBossSpall.mjd < mjd_end)

        if self._query_filter_by_kwargs is not None:
            q = q.filter_by(**self._query_filter_by_kwargs)

        if self._limit is not None:
            q = q.limit(self._limit)

        log.debug(f"Found {q.count()} {release} {filetype} files between MJD {mjd_start} and {mjd_end}")

        common = dict(release=release, filetype=filetype)
        keys = [column.name for column in columns]
        for values in q.yield_per(1):
            yield { **common, **dict(zip(keys, values)) }

