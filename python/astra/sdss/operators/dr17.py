""" Operators for ingesting data products from SDSS-IV data release 17. """

from itertools import starmap
import os
from astra import log, config
from astra.database.astradb import DataProduct, Source, SourceDataProduct
from astropy.time import Time
from airflow.exceptions import AirflowSkipException
from peewee import fn
from typing import Optional, List

from astra.sdss.operators.base import SDSSOperator
from astra.utils import flatten, expand_path


class ApogeeOperator(SDSSOperator):
    def __init__(
        self,
        return_id_kind: Optional[str] = "data_product",
        apogee_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super(ApogeeOperator, self).__init__(**kwargs)
        self._available_return_id_kinds = ["catalog", "data_product"]
        if return_id_kind not in self._available_return_id_kinds:
            raise ValueError(
                f"return_id_kind must be one of {self._available_return_id_kinds}"
            )
        self.return_id_kind = return_id_kind
        self.apogee_ids = apogee_ids
        return None

class SelectApStarOperator(ApogeeOperator):
    release = "dr17"

    def execute(self, context: dict):
        """
        Execute this operator.

        :param context:
            The Airflow context dictionary.
        """
        q = (
            DataProduct
            .select(DataProduct.id)
            .where(
                (DataProduct.filetype.in_(("apStar", "apStar-1m")))
            &   (DataProduct.release == self.release)
            )
        )
        if self.apogee_ids is not None:
            if isinstance(self.apogee_ids, str) and os.path.exists(expand_path(self.apogee_ids)):
                with open(expand_path(self.apogee_ids), "r") as fp:
                    apogee_ids = list(map(str.strip, fp.readlines()))
            else:
                apogee_ids = self.apogee_ids 
            log.info(f"Restricting to {len(apogee_ids)} APOGEE_ID: {apogee_ids}")
            q = q.where(DataProduct.kwargs["obj"].in_(apogee_ids))

        assert self.return_id_kind == "data_product"

        ids = flatten(q.tuples())
        log.info(f"Found {len(ids)} apStar data products.")
        return ids

    

class ApStarOperator(ApogeeOperator):

    """An operator to retrieve and ingest ApStar data products created in SDSS-IV."""

    release = "dr17"

    def filetype(self, telescope):
        """
        Return the file type used by ``sdss_access``, given the telescope.

        For historical reasons, a different data product was used for observations taken with the 1-meter.

        :param telescope:
            A string describing the telescope used (e.g., apo25m, lco25m, apo1m)
        """
        return "apStar-1m" if telescope == "apo1m" else "apStar"

    def execute(self, context: dict):
        """
        Execute this operator.

        :param context:
            The Airflow context dictionary.
        """
        ds = context.get("ds", None)
        prev_ds = context.get("prev_ds", None)

        from astra.database.catalogdb import (
            Catalog,
            # CatalogToGaia_DR3,
            Gaia_DR3,
            SDSS_DR17_APOGEE_Allstarmerge as Star,
            #SDSS_DR17_APOGEE_Allvisits as Visit
            # SDSS_DR17_APOGEE_Allvisits as Visit
        )
        try:
            from astra.database.catalogdb import SDSS_DR17_APOGEE_Allvisits
        except ImportError:
            from astra.database.catalogdb import CatalogdbModel, TextField

            # TODO: this table not yet in sdssdb, will be in next version
            #       remove this after sdssdb 0.5.4
            class SDSS_DR17_APOGEE_Allvisits(CatalogdbModel):

                visit_id = TextField(primary_key=True)

                class Meta:
                    table_name = "sdss_dr17_apogee_allvisits"

            Visit = SDSS_DR17_APOGEE_Allvisits

        else:
            Visit = SDSS_DR17_APOGEE_Allvisits
            print(f"ANDY: remove SDSS_DR17_APOGEE_Allvisits in {__file__}")        

        # I first thought I'd have to query Star, matched on Visit, but actually the Star merge table
        # munges columns like 'telescope' -> 'telescopes' (all visits) and ignores things like field.
        # That means the Star merge table is nearly useless for our purposes here (we only use it to
        # get gaia_source_id and cross-match to the catalog).

        # When catalog_to_sdss_dr17_apogee_allvisits exists, we won't need to cross-match via Stars at all
        fields = (
            Visit.ra,
            Visit.dec,
            Visit.apogee_id,
            Visit.telescope,
            Visit.field,
            Star.gaia_source_id,
        )
        q = (
            Visit.select(*fields)
            .distinct()
            .join(Star, on=(Star.apogee_id == Visit.apogee_id))
        )
        if self.apogee_ids is not None:
            log.info(f"Restricting to APOGEE_ID: {self.apogee_ids}")
            q = q.where(Visit.apogee_id.in_(self.apogee_ids))

        if ds is not None and prev_ds is not None:
            # Only retrieve stars by their most recent MJD, which requires an internal match on Visit
            q = q.group_by(*fields).having(
                fn.MAX(Visit.mjd).between(Time(prev_ds).mjd, Time(ds).mjd)
            )

        ids, errors = ([], [])
        for star in q:
            # The database table column names do not match the path definition.
            # Thankfully, neither will change so we can hard-code this fix in.
            kwds = {
                # TODO: If this screws up, we may need to select Visit.file from the
                #       sub-query and get the prefix from the first two characters
                "prefix": "as" if star.telescope == "lco25m" else "ap",
                "apstar": "stars",
                "apred": self.release,
                "field": star.field.strip(),
                "telescope": star.telescope,
                "obj": star.apogee_id,
            }
            if self.filetype(star.telescope) == "apStar-1m":
                # The 'reduction' keyword is used by apStar-1m, but not apStar. No idea why it's not ``obj``..
                kwds.update(reduction=star.apogee_id)

            path = self.path_instance.full(self.filetype(star.telescope), **kwds)
            if not os.path.exists(path):
                errors.append({
                    "detail": path, 
                    "origin": star, 
                    "reason": "File does not exist",
                    "kwds": kwds
                })
                log.warning(
                    f"Error ingesting path {path} from {star}:\n\t{errors[-1]['reason']}\nwith keywords {kwds}"
                )
                continue
            else:
                # Cross-match to catalogdb.
                """
                (catalogid,) = (
                    Catalog.select(Catalog.catalogid)
                    .join(CatalogToGaia_DR3)
                    .join(Gaia_DR3)
                    .where(Gaia_DR3.source_id == star.gaia_source_id)
                    .tuples()
                    .first()
                )
                """
                try:
                    (catalogid,) = (
                        Catalog.select(Catalog.catalogid)
                        .where(Catalog.cone_search(star.ra, star.dec, 3.0 / 3600))
                        .tuples()
                        .first()
                    )
                except:
                    log.warning(
                        f"No Catalog entry found at ra={star.ra}, dec={star.dec}. Skipping {path} from {star}"
                    )
                    errors.append(
                        {
                            "detail": path,
                            "origin": star,
                            "reason": "No source found in catalog.",
                            "kwds": kwds,
                        }
                    )
                    continue

                source, _ = Source.get_or_create(catalogid=catalogid)
                data_product, created = DataProduct.get_or_create(
                    release=self.release,
                    filetype=self.filetype(star.telescope),
                    kwargs=kwds,
                )
                SourceDataProduct.get_or_create(
                    source=source, data_product=data_product
                )
                ids.append((catalogid, data_product.id))
                log.info(f"Created data product {data_product} for source {source}")

        log.info(f"Ingested {len(ids)} ApStar products")
        log.info(f"Encountered {len(errors)} errors.")
        if len(errors) > 0:
            for error in errors:
                log.warning(
                    f"Error ingesting path {error['detail']} from {error['origin']}:\n\t{error['reason']}\nwith keywords {error['kwds']}"
                )
        if len(ids) == 0:
            raise AirflowSkipException(f"No data products ingested.")

        index = self._available_return_id_kinds.index(self.return_id_kind)
        return list(set([each[index] for each in ids]))


class ApVisitOperator(ApogeeOperator):

    """An operator to retrieve and ingest ApVisit data products created in SDSS-IV."""

    release = "dr17"
    filetype = "apVisit"

    def execute(self, context: dict):
        """
        Execute this operator.

        :param context:
            The Airflow context dictionary.
        """

        from astra.database.catalogdb import (
            Catalog,
            # CatalogToGaia_DR3,
            Gaia_DR3,
            SDSS_DR17_APOGEE_Allstarmerge as Star,
            SDSS_DR17_APOGEE_Allvisits as Visit
            # SDSS_DR17_APOGEE_Allvisits as Visit
        )
        ds = context.get("ds", None)
        prev_ds = context.get("prev_ds", None)

        fields = (
            Visit.ra,
            Visit.dec,
            Visit.visit_id,
            Visit.apogee_id,
            Visit.mjd,
            Visit.plate,
            Visit.telescope,
            Visit.fiberid,
            Visit.field,
        )
        if ds is not None and prev_ds is not None:
            # Doing a direct query between Visit and Star causes a sequential scan across Star, which is really
            # slow. Instead let's try a sub-query.
            if self.apogee_ids is not None:
                log.info(f"Restricting to APOGEE_ID: {self.apogee_ids}")
                sq = (
                    Visit.select(*fields)
                    .where(Visit.apogee_id.in_(self.apogee_ids))
                    .where(Visit.mjd.between(Time(prev_ds).mjd, Time(ds).mjd))
                    .alias("sq")
                )
            else:
                sq = (
                    Visit.select(*fields)
                    .where(Visit.mjd.between(Time(prev_ds).mjd, Time(ds).mjd))
                    .alias("sq")
                )
            q = (
                Star.select(
                    Star.gaia_source_id,
                    sq.c.ra,
                    sq.c.dec,
                    sq.c.visit_id,
                    sq.c.apogee_id,
                    sq.c.mjd,
                    sq.c.plate,
                    sq.c.telescope,
                    sq.c.fiberid,
                    sq.c.field,
                )
                .join(sq, on=(Star.apogee_id == sq.c.apogee_id))
                .objects()
            )

        else:
            # When catalog_to_sdss_dr17_apogee_allvisits exists, we won't need to cross-match via Stars at all
            # This will do a sequential scan across Star, its slow..
            q = Visit.select(Star.gaia_source_id, *fields).join(
                Star, on=(Star.apogee_id == Visit.apogee_id)
            )
            if self.apogee_ids is not None:
                log.info(f"Restricting to APOGEE_ID: {self.apogee_ids}")
                q = q.where(Visit.apogee_id.in_(self.apogee_ids))

            q = q.objects()

        ids, errors = ([], [])
        for visit in q:
            # The database table column names do not match the path definition.
            # Thankfully, neither will change so we can hard-code this fix in.
            kwds = {
                # TODO: If this screws up, we may need to select Visit.file from the
                #       sub-query and get the prefix from the first two characters
                "mjd": visit.mjd,
                "plate": visit.plate.strip(),
                "apred": self.release,
                "prefix": "as" if visit.telescope == "lco25m" else "ap",
                "fiber": visit.fiberid,
                "field": visit.field.strip(),
                "telescope": visit.telescope,
            }
            path = self.path_instance.full(self.filetype, **kwds)
            if not os.path.exists(path):
                errors.append(
                    {
                        "detail": path,
                        "origin": visit,
                        "reason": "File does not exist",
                        "kwds": kwds,
                    }
                )
                log.warning(
                    f"Error ingesting path {path} from {visit}:\n\t{errors[-1]['reason']}\nwith keywords {kwds}"
                )
                continue
            else:
                # Cross-match to catalogdb.
                """
                (catalogid,) = (
                    Catalog.select(Catalog.catalogid)
                    .join(CatalogToGaia_DR3)
                    .join(Gaia_DR3)
                    .where(Gaia_DR3.source_id == visit.gaia_source_id)
                    .tuples()
                    .first()
                )
                """
                try:
                    (catalogid,) = (
                        Catalog.select(Catalog.catalogid)
                        .where(Catalog.cone_search(visit.ra, visit.dec, 3.0 / 3600))
                        .tuples()
                        .first()
                    )
                except:
                    log.warning(
                        f"No Catalog entry found at ra={visit.ra}, dec={visit.dec}. Skipping {path} from {visit}"
                    )
                    errors.append(
                        {
                            "detail": path,
                            "origin": visit,
                            "reason": "No source found in catalog.",
                            "kwds": kwds,
                        }
                    )
                    continue

                source, _ = Source.get_or_create(catalogid=catalogid)
                data_product, created = DataProduct.get_or_create(
                    release=self.release,
                    filetype=self.filetype,
                    kwargs=kwds,
                )
                SourceDataProduct.get_or_create(
                    source=source, data_product=data_product
                )
                ids.append((catalogid, data_product.id))
                log.info(f"Created data product {data_product} for source {source}")

        log.info(f"Ingested {len(ids)} ApStar products")
        log.info(f"Encountered {len(errors)} errors.")
        if len(errors) > 0:
            for error in errors:
                log.warning(
                    f"Error ingesting path {error['path']} from {error['origin']}:\n\t{error['reason']}\nwith keywords {error['kwds']}"
                )

        if len(ids) == 0:
            raise AirflowSkipException(f"No data products ingested.")

        index = self._available_return_id_kinds.index(self.return_id_kind)
        return list(set([each[index] for each in ids]))
