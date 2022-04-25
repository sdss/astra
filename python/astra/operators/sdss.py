import os
from typing import OrderedDict
from astropy.io import fits
from airflow.models.baseoperator import BaseOperator
from airflow.exceptions import (AirflowFailException, AirflowSkipException)
from sdss_access import SDSSPath
from astra.database.apogee_drpdb import Star, Visit
from astra.database.catalogdb import SDSSVBossSpall
from astra.database.astradb import (database, DataProduct, Source, SourceDataProduct)
from astra import log
from functools import lru_cache

from peewee import fn

@lru_cache
def path_instance(release):
    return SDSSPath(release=release)


@lru_cache
def lookup_keys(release, filetype):
    return path_instance(release).lookup_keys(filetype)

from time import time
import numpy as np
def get_or_create_data_product_from_apogee_drpdb(
        origin, 
        release=None,
        filetype=None,
    ):
    """
    Get or create a data product entry in the astra database from an apogee_drpdb origin.
    
    :param release: [optional]
        Supply a release. Otherwise this will be read from `origin.release`.

    :param filetype: [optional]
        The filetype of the data product. If `None`, this will be read from `origin.filetype`.
    """
    t_init = time()

    release = release or origin.release
    filetype = filetype or origin.filetype

    kwds = { k: getattr(origin, k) for k in lookup_keys(release, filetype) }
    if "field" in kwds:
        kwds["field"] = kwds["field"].strip()
    t_kwds = time() - t_init
    t_init = time()
    path = path_instance(release).full(filetype, **kwds)
    t_path = time() - t_init

    t_init = time()
    if not os.path.exists(path):
        error = {
            "detail": path,
            "origin": origin,
            "reason": "File does not exist"
        }        
        return (False, error)
    t_check_path = time() - t_init

    # Try to get a size.
    try:
        size = origin.nvisits # apogee_drp.star
    except:
        size = 1

    # TODO: If we the data product already exists, check that the size matches
    with database.atomic() as txn:
        t_init = time()
        data_product, data_product_created = DataProduct.get_or_create(
            release=release,
            filetype=filetype,
            kwargs=kwds,
            defaults=dict(size=size)
        )
        if not data_product_created:
            # Update the size
            if data_product.size != size:
                log.info(f"Updating size of data product {data_product} from {data_product.size} to {size}")
                data_product.size = size
                data_product.save()

        t_dp = time() - t_init
        t_init = time()
        source, _ = Source.get_or_create(catalogid=origin.catalogid)
        t_source = time() - t_init
        t_init = time()
        SourceDataProduct.get_or_create(
            source=source,
            data_product=data_product
        )
        t_sourcedataproduct = time() - t_init

    ts = np.array([t_kwds, t_path, t_check_path, t_dp, t_source, t_sourcedataproduct])
    #print(np.round(ts/np.sum(ts), 1))

    result = dict(data_product=data_product, source=source)
    return (True, result)


class BossSpecOperator(BaseOperator):
    """
    A base operator for working with SDSS-V BOSS spectrum data products. 
    
    This operator will generate task instances based on BOSS spec data products it finds that were
    *observed* in the operator execution period.
    """


    ui_color = "#A0B9D9"
    
    def execute(self, context):
        raise NotImplementedError("spec still seems to not be in the tree product")


    def query_data_model_identifiers_from_database(self, context):
        """
        Query the SDSS-V database for BOSS spectrum data model identifiers.

        :param context:
            The Airflow DAG execution context.
        """ 

        release, filetype = ("SDSS5", "spec")
        
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






class ApVisitOperator(BaseOperator):

    def execute(self, context):

        model, prev_ds, ds = (Visit, context["prev_ds"], context["ds"])
        where = (Visit.catalogid > 0)
    
        log.info(f"{self} is looking for products created between {prev_ds} and {ds}")
        q = (
            model.select()
                 .where(model.created.between(prev_ds, ds))
        )
        if where is not None:
            q = q.where(where)

        N = q.count()
        if N == 0:
            raise AirflowSkipException(f"No products for {self} created between {prev_ds} and {ds}")

        ids = []
        errors = []
        for origin in q:
            success, result = get_or_create_data_product_from_apogee_drpdb(origin)
            if success:
                log.info("Data product {data_product} matched to {source}".format(**result))
                ids.append(result["data_product"].id)
            else:
                log.warning("{reason} ({detail}) from {origin}".format(**result))
                errors.append(result)

        log.info(f"Found {N} rows.")
        log.info(f"Created {len(ids)} data products.")
        log.info(f"Encountered {len(errors)} errors.")
        if len(errors) == N:
            raise AirflowFailException(f"{N}/{N} data products had errors")
        return ids



class ApStarOperator(BaseOperator):
    """
    Generate data model products in the Astra database for all new ApStar
    files produced since the operator was last executed.

    :param releases: [optional]
        The relevant SDSS data releases. If `None` is given then this will be inferred based on
        the execution date.
    """

    ui_color = "#ffb09c"

    def execute(
        self, 
        context,
        where=(Star.ngoodvisits > 0) & (Star.catalogid > 0),
        latest_only=True
    ):
        prev_ds, ds = (context["prev_ds"], context["ds"])
    
        log.info(f"{self} is looking for {'only latest' if latest_only else 'any'} ApStar products created between {prev_ds} and {ds}")

        if latest_only:
            StarAlias = Star.alias()
            sq = (
                StarAlias.select(
                    StarAlias.catalogid, 
                    fn.MAX(StarAlias.created).alias("max_created")
                ).group_by(StarAlias.catalogid).alias("sq")
            )
            q = (
                Star.select()
                    .where(
                        sq.c.max_created.between(prev_ds, ds)
                        & where
                    )
                    .join(
                        sq, 
                        on=(Star.created == sq.c.max_created) 
                         & (Star.catalogid == sq.c.catalogid)
                    )
            )
        else:
            q = (
                Star.select()
                    .where(
                        Star.created.between(prev_ds, ds)
                        & where
                    )
            )

        N = q.count()
        if N == 0:
            raise AirflowSkipException(f"No products for {self} created between {prev_ds} and {ds}")

        ids = []
        errors = []
        for origin in q:
            success, result = get_or_create_data_product_from_apogee_drpdb(origin)
            if success:
                log.info("Data product {data_product} matched to {source}".format(**result))
                ids.append(result["data_product"].id)
            else:
                log.warning("{reason} ({detail}) from {origin}".format(**result))
                errors.append(result)

        log.info(f"Found {N} rows.")
        log.info(f"Created {len(ids)} data products.")
        log.info(f"Encountered {len(errors)} errors.")
        if len(errors) == N:
            raise AirflowFailException(f"{N}/{N} data products had errors")
        return ids


