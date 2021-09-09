
from sqlalchemy import and_, func

from astra.utils import log

from astra.database import (apogee_drpdb, catalogdb, session)
from astra.operators.sdss_data_product import DataProductOperator
from astra.operators.utils import parse_as_mjd

class ApStarOperator(DataProductOperator):
    """
    A base operator for working with SDSS ApStar data products. 
    
    This operator will generate task instances based on ApStar data products it finds that were
    completed in the operator execution period.

    :param release: [optional]
        The relevant SDSS data release. If `None` is given then this will be inferred based on
        the execution date.
    :type release: str
    :param _limit: [optional]
        Limit the database query to only return `_limit` ApStar objects.
    :type _limit: int
    :param _skip_sources_with_more_recent_observations: [optional]
        Skip any ApStar sources that have observations more recent than the current execution
        date. The ApStar data products include spectra from multiple individual visits. If we
        had an execution date of 01-01-2021 and a source was observed with an APOGEE instrument
        on 01-01-2021, then it would be included with this operator's execution. However, if
        we are back-filling analyses (e.g., catching up on analysis) and we see that this source
        was later observed on 02-02-2021, then we would be executing the analysis twice on
        exactly the same spectra: 01-01-2021 and 02-02-2021 (since the ApStar object contains
        the information for both observations).

        If we set `_skip_sources_with_more_recent_observations` as True (default), then we
        will only analyse spectra where the *final* observation (to date) has occurred in the
        execution period. That gives the expected behaviour for daily analyses (e.g., new 
        observations just taken will be analysed), and when back-filling analyses we will not
        be re-analysing the same spectra N times.
    :type _skip_sources_with_more_recent_observations: bool
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

    ui_color = "#ffb09c"

    def __init__(
        self,
        *,
        release = None,
        # We want to be able to supply these arguments, but we don't want them stored as parameters
        # in the task instances, so we prefix them with '_'.
        _limit = None,
        _skip_sources_with_more_recent_observations = True,
        _query_filter_by_kwargs = None,
        **kwargs,
    ) -> None:
        super(ApStarOperator, self).__init__(**kwargs)
        self.release = release
        self._limit = _limit
        self._query_filter_by_kwargs = _query_filter_by_kwargs
        self._skip_sources_with_more_recent_observations = _skip_sources_with_more_recent_observations
    

    def query_sdss4_dr16_data_model_identifiers_from_database(self, mjd_start, mjd_end):
        """
        Query the SDSS database for SDSS-IV (DR16) ApStar data model identifiers.

        :param context:
            The Airflow DAG execution context.
        """

        release, filetype = ("DR16", "apStar")
        columns = (
            func.left(catalogdb.SDSSDR16ApogeeStar.file, 2).label("prefix"),
            catalogdb.SDSSDR16ApogeeStar.field,
            catalogdb.SDSSDR16ApogeeStar.apstar_version.label("apstar"),
            catalogdb.SDSSDR16ApogeeStar.telescope,
            catalogdb.SDSSDR16ApogeeStar.apogee_id.label("obj"),
            func.right(func.left(catalogdb.SDSSDR16ApogeeStar.file, 10), 3).label("apred"),
        )

        if not self._skip_sources_with_more_recent_observations:
            # The SDSSDR16ApogeeStar table does not have any MJD information.
            mjd = catalogdb.SDSSDR16ApogeeVisit.mjd
            q = session.query(*columns, mjd).distinct(*columns, mjd).join(
                catalogdb.SDSSDR16ApogeeVisit,
                catalogdb.SDSSDR16ApogeeVisit.apogee_id == catalogdb.SDSSDR16ApogeeStar.apogee_id
            )
        
        else:
            # Get the max MJD of any observations.
            sq = session.query(
                *columns, 
                func.max(catalogdb.SDSSDR16ApogeeVisit.mjd).label('max_mjd')
            ).join(
                catalogdb.SDSSDR16ApogeeVisit,
                catalogdb.SDSSDR16ApogeeVisit.apogee_id == catalogdb.SDSSDR16ApogeeStar.apogee_id
            ).group_by(*columns).subquery()

            mjd = sq.c.max_mjd
            q = session.query(*columns, mjd).join(
                sq,
                catalogdb.SDSSDR16ApogeeStar.apogee_id == sq.c.obj
            )

        q = q.filter(mjd < mjd_end)\
             .filter(mjd >= mjd_start)

        if self._query_filter_by_kwargs is not None:
            q = q.filter_by(**self._query_filter_by_kwargs)

        if self._limit is not None:
            q = q.limit(self._limit)

        log.debug(f"Found {q.count()} {release} {filetype} files between MJD {mjd_start} and {mjd_end}")

        common = dict(release=release, filetype=filetype)
        keys = [column.name for column in columns]
        # The MJD will not be included because len(keys) < len(values) and zip will only take the shorter of both.
        for values in q.yield_per(1):
            yield { **common, **dict(zip(keys, values)) }
                        

    def query_sdss5_data_model_identifiers_from_database(self, mjd_start, mjd_end):
        """
        Query the SDSS-V database for ApStar data model identifiers.
        """
        release, filetype, apstar = ("sdss5", "apStar", "stars")
        
        columns = (
            apogee_drpdb.Star.apred_vers.label("apred"), # TODO: Raise with Nidever
            apogee_drpdb.Star.healpix,
            apogee_drpdb.Star.telescope,
            apogee_drpdb.Star.apogee_id.label("obj"), # TODO: Raise with Nidever
        )
        
        if not self._skip_sources_with_more_recent_observations:
            q = session.query(*columns).distinct(*columns)
        else:
            # Get the max MJD of any observations for this source.
            sq = session.query(
                *columns, 
                func.max(apogee_drpdb.Star.mjdend).label('max_mjdend')
            ).group_by(*columns).subquery()

            q = session.query(*columns).join(
                sq, 
                and_(
                    apogee_drpdb.Star.mjdend == sq.c.max_mjdend,
                    apogee_drpdb.Star.apred_vers == sq.c.apred,
                    apogee_drpdb.Star.healpix == sq.c.healpix,
                    apogee_drpdb.Star.telescope == sq.c.telescope,
                    apogee_drpdb.Star.apogee_id == sq.c.obj
                )        
            )
        
        # Filter on number of good RV measurements, and the MJD of last obs 
        q = q.filter(apogee_drpdb.Star.mjdend < mjd_end)\
             .filter(apogee_drpdb.Star.mjdend >= mjd_start)

        if self._query_filter_by_kwargs is not None:
            q = q.filter_by(**self._query_filter_by_kwargs)

        if self._limit is not None:
            q = q.limit(self._limit)

        log.debug(f"Preparing query {q}")
        total = q.count()
        log.debug(f"Retrieved {total} rows between {mjd_start} <= MJD < {mjd_end}")


        keys = [column.name for column in columns]

        for values in q.yield_per(1):
            d = dict(zip(keys, values))
            d.update(
                release=release,
                filetype=filetype,
                apstar=apstar,
            )
            yield d