import datetime
import os
from typing import OrderedDict, Optional
from astropy.io import fits
from airflow.models.baseoperator import BaseOperator
from airflow.exceptions import AirflowFailException, AirflowSkipException
from sdss_access import SDSSPath
from astra.database.apogee_drpdb import Star, Visit
from astra.database.astradb import database, DataProduct, Source
from astra.utils import log, flatten

from functools import lru_cache

from peewee import fn
from tqdm import tqdm

class ApVisitOperator(BaseOperator):
    def __init__(
        self,
        apred: Optional[str] = "daily",
        check_path: Optional[bool] = True,
        minimum_snr: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.apred = apred
        self.check_path = check_path
        self.minimum_snr = minimum_snr
        return None


    def execute(self, context):

        release, filetype = ("sdss5", "apVisit")
        prev_ds, ds = (context["prev_ds"], context["ds"])

        # catalogid < 0 apparently implies unassigned fibre
        # TODO: check if they are really unassigned ornot .
        # apogee-pipeline 470] off target robots in reductions

        # Through inspection I found that all the visit (Database) records with missing paths
        # tended to be those without any sdssv_apogee_target0 value.
        #'telescope', 'plate', 'field', 'mjd', 'fiber', 'apred'
        log.info(f"{self} is looking for products created between {prev_ds} and {ds} (check_path={self.check_path})")

        q = (
            Visit.select(
                Visit.catalogid,
                Visit.ra,
                Visit.dec,
                Visit.telescope,
                Visit.plate,
                Visit.field,
                Visit.mjd,
                Visit.fiber,
                Visit.apred,
            )
            .where(
                (Visit.apred_vers == self.apred) & (Visit.catalogid > 0)
            )
        )
        if prev_ds is not None:
            q = q.created.between(prev_ds, ds)
        
        if self.minimum_snr is not None:
            q = q.where(Visit.snr > self.minimum_snr)

        q = q.order_by(Visit.pk.asc()).tuples()

        N = q.count()
        if N == 0:
            raise AirflowSkipException(
                f"No products for {self} created between {prev_ds} and {ds}"
            )

        log.info(f"Found {N} rows with Visit.apred_vers = {self.apred} and Visit.catalogid > 0")

        p = SDSSPath(release)
        keys = ("telescope", "plate", "field", "mjd", "fiber", "apred")    
        catalogids, data_product_kwds, errors = ([], [], [])
        for N, (catalogid, ra, dec, *values) in enumerate(tqdm(q, total=N, desc="Preparing rows"), start=1):
            kwargs = dict(zip(keys, values))
            kwargs["field"] = kwargs["field"].strip()
            if not isinstance(catalogid, int):
                error = {"detail": None, "origin": (catalogid, *values), "reason": f"Catalogid is not an integer ({type(catalogid)}: {catalogid})" }
                errors.append(error)
                continue
        
            if self.check_path:
                path = p.full(filetype, **kwargs)
                if not os.path.exists(path):
                    error = {"detail": path, "origin": (catalogid, *values), "reason": "File does not exist"}
                    errors.append(error)
                    continue

            source, _ = Source.get_or_create(
                catalogid=catalogid,
                defaults=dict(ra=ra, dec=dec)
            )
            kwargs, kwargs_hash = DataProduct.adapt_and_hash_kwargs(kwargs)
            catalogids.append(catalogid)
            data_product_kwds.append(
                dict(
                    release=release,
                    filetype=filetype,
                    kwargs=kwargs,
                    source=source,
                    kwargs_hash=kwargs_hash
                )
            )
        
        N -= len(errors)

        log.info(f"Doing bulk upserts..")
        with database.atomic():
            #log.info(f"Creating {N} sources..")
            #Source.insert_many([dict(catalogid=catalogid) for catalogid in catalogids]).on_conflict_ignore().execute()

            log.info(f"Creating {N} data products..")
            data_product_ids = flatten(
                DataProduct
                .insert_many(data_product_kwds)
                .returning(DataProduct.id)
                # If we use on_conflict_ignore() then we only get IDs for the new rows.
                .on_conflict(
                    conflict_target=[DataProduct.release, DataProduct.filetype, DataProduct.kwargs_hash],
                    update={
                        DataProduct.updated: datetime.datetime.now()
                    }
                ) 
                .tuples()
                .execute()
            )
            assert len(catalogids) == len(data_product_ids)
            
        log.info(f"Done!")
        
        log.info(f"Created {len(data_product_ids)} data products.")
        if self.check_path:
            log.warning(f"Encountered {len(errors)} errors.")
            for error in errors:
                log.warning("{reason}: {detail} from {origin}".format(**error))
        else:
            log.warning(f"Did not check paths. Some data products may not exist on disk.")
        
        if len(errors) == N:
            raise AirflowSkipException(f"{N}/{N} data products had errors")
        return data_product_ids


class ApStarOperator(BaseOperator):
    """
    Generate data model products in the Astra database for all new ApStar
    files produced since the operator was last executed.
    """

    ui_color = "#ffb09c"

    def execute(
        self,
        context,
        latest_only=True,
    ):
        where = (Star.ngoodvisits > 0) & (Star.catalogid > 0)
        prev_ds, ds = (context["prev_ds"], context["ds"])

        log.info(
            f"{self} is looking for {'only latest' if latest_only else 'any'} ApStar products created between {prev_ds} and {ds}"
        )

        if latest_only:
            StarAlias = Star.alias()
            sq = (
                StarAlias.select(
                    StarAlias.catalogid, fn.MAX(StarAlias.created).alias("max_created")
                )
                .group_by(StarAlias.catalogid)
                .alias("sq")
            )
            q = (
                Star.select()
                .where(sq.c.max_created.between(prev_ds, ds) & where)
                .join(
                    sq,
                    on=(Star.created == sq.c.max_created)
                    & (Star.catalogid == sq.c.catalogid),
                )
            )
        else:
            q = Star.select().where(Star.created.between(prev_ds, ds) & where)

        N = q.count()
        if N == 0:
            raise AirflowSkipException(
                f"No products for {self} created between {prev_ds} and {ds}"
            )

        ids = []
        errors = []
        for origin in q:
            raise NotUpdatedErrorYet
            success, result = get_or_create_data_product_from_apogee_drpdb(origin)
            if success:
                log.info(
                    "Data product {data_product_id} matched to {source_id}".format(
                        **result
                    )
                )
                ids.append(result["data_product_id"])
            else:
                log.warning("{reason} ({detail}) from {origin}".format(**result))
                errors.append(result)

        log.info(f"Found {N} rows.")
        log.info(f"Created {len(ids)} data products.")
        log.info(f"Encountered {len(errors)} errors.")
        if len(errors) == N:
            raise AirflowFailException(f"{N}/{N} data products had errors")
        return ids


@lru_cache
def path_instance(release):
    return SDSSPath(release=release)


@lru_cache
def lookup_keys(release, filetype):
    return path_instance(release).lookup_keys(filetype)


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
    release = release or origin.release
    filetype = filetype or origin.filetype

    kwds = {k: getattr(origin, k) for k in lookup_keys(release, filetype)}
    if "field" in kwds:
        kwds["field"] = kwds["field"].strip()
    path = path_instance(release).full(filetype, **kwds)
    if not os.path.exists(path):
        error = {"detail": path, "origin": origin, "reason": "File does not exist"}
        return (False, error)

    # TODO: If we the data product already exists, check that the size matches
    # with database.atomic() as txn:
    # Have to make sure the source exists before we link SourceDataProduct..
    source, _ = Source.get_or_create(catalogid=origin.catalogid)

    data_product, data_product_created = DataProduct.get_or_create(
        release=release,
        filetype=filetype,
        source=source,
        kwargs=kwds,
    )
    # if not data_product_created:
    #    # Update the size
    #    if data_product.size != size:
    #        log.info(f"Updating size of data product {data_product} from {data_product.size} to {size}")
    #        data_product.size = size
    #        data_product.save()


    result = dict(data_product_id=data_product.id, source_id=origin.catalogid)
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
            visit = (
                Visit.select()
                .where(
                    (Visit.apogee_id == apogee_id)
                    & (Visit.dateobs == image[0].header[f"DATE{i:.0f}"])
                )
                .first()
            )

            visit_pk = None if visit is None else visit.pk
            meta.append(
                {
                    "visit_pk": visit_pk,
                    "fiber": image[0].header[f"FIBER{i:.0f}"],
                    "date_obs": image[0].header[f"DATE{i:.0f}"],
                    "jd": image[0].header[f"JD{i:.0f}"],
                    "bc": image[0].header[f"BC{i:.0f}"],
                    "vrad": image[0].header[f"VRAD{i:.0f}"],
                    "vhbary": image[0].header[f"VHBARY{i:.0f}"],
                    # 2022-04-28: STARFLAG is not present in all ApStar files
                    # "starflag": image[0].header[f"STARFLAG{i:.0f}"]
                }
            )

    return meta
