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
from astropy.time import Time
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


def get_apvisit_metadata(apstar_data_product):
    # It's stupid to open the file for this, but I can't yet seem to find the magic to
    # match visits to apstar files in apogee_drpdb because visits are excluded from the
    # stack for a variety of reasons.
    apogee_id = apstar_data_product.kwargs["obj"]
    meta = []
    with fits.open(apstar_data_product.path) as image:
        N, P = image[1].data.shape
        if N > 2:
            meta.extend([{"note": "stacked_0"}, {"note": "stacked_1"}])
            indices = range(1, N - 1)
        else:
            indices = [1]

        for i in indices:
            # get visit pk
            visit = Visit.select().where(
                    (Visit.apogee_id == apogee_id)
                &   (Visit.dateobs == image[0].header[f"DATE{i:.0f}"])
            ).first()
            
            visit_pk = None if visit is None else visit.pk
            meta.append({
                "visit_pk": visit_pk,
                "fiber": image[0].header[f"FIBER{i:.0f}"],
                "date_obs": image[0].header[f"DATE{i:.0f}"],
                "jd": image[0].header[f"JD{i:.0f}"],
                "bc": image[0].header[f"BC{i:.0f}"],
                "vrad": image[0].header[f"VRAD{i:.0f}"],
                "vhbary": image[0].header[f"VHBARY{i:.0f}"],
                # 2022-04-28: STARFLAG is not present in all ApStar files
                #"starflag": image[0].header[f"STARFLAG{i:.0f}"]
            })
    
    return meta



class BossSpecLiteOperator(BaseOperator):
    """
    A base operator for working with SDSS-V BOSS spectrum data products. 
    
    This operator will generate task instances based on BOSS spec data products it finds that were
    *observed* in the operator execution period.
    """

    ui_color = "#A0B9D9"

    def execute(self, context):
        release, filetype = ("sdss5", "specLite")

        prev_ds, ds = (context["prev_ds"], context["ds"])
        mjd_start, mjd_end = list(map(lambda x: Time(x).mjd, (prev_ds, ds)))

        # Unbelievably, this is still not stored in the database.
        from astropy.table import Table
        from astra.utils import expand_path
        data = Table.read(expand_path("$BOSS_SPECTRO_REDUX/master/spAll-master.fits"))
        is_mwm = np.array([pg.strip().startswith("mwm_") or fc.strip().startswith("mwm_") for pg, fc in zip(data["PROGRAMNAME"], data["FIRSTCARTON"])])
        in_mjd_range = (mjd_end > data["MJD"]) & (data["MJD"] >= mjd_start)
        mask = is_mwm * in_mjd_range

        N = sum(mask)
        if N == 0:
            raise AirflowSkipException(f"No products for {self} created between {prev_ds} and {ds}")

        errors = []
        ids = []
        for row in data[mask]:
            catalogid = row["CATALOGID"]
            kwds = dict(
                # TODO: remove this when the path is fixed in sdss_access
                fieldid=f"{row['FIELD']:0>6.0f}",
                mjd=int(row["MJD"]),
                catalogid=int(catalogid),
                run2d=row["RUN2D"],
                isplate=""
            )
            path = path_instance(release).full(filetype, **kwds)
            if not os.path.exists(path):
                error = {
                    "detail": path,
                    "origin": row,
                    "reason": "File does not exist"
                }
                errors.append(error)
                continue
                
            with database.atomic() as txn:
                data_product, _ = DataProduct.get_or_create(
                    release=release,
                    filetype=filetype,
                    kwargs=kwds,
                )
                source, _ = Source.get_or_create(catalogid=catalogid)
                SourceDataProduct.get_or_create(
                    source=source,
                    data_product=data_product
                )
            
            log.info(f"Data product {data_product} matched to {source}")
            ids.append(data_product.id)

        N = sum(mask)
        log.info(f"Found {N} rows.")
        log.info(f"Created {len(ids)} data products.")
        log.info(f"Encountered {len(errors)} errors.")
        if len(errors) == N:
            raise AirflowSkipException(f"{N}/{N} data products had errors")
        return ids




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
            raise AirflowSkipException(f"{N}/{N} data products had errors")
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


