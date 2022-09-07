""" Operators for ingesting data products from SDSS-IV data release 17. """

import os
from astra import log, config
from astra.database.astradb import DataProduct, Source, SourceDataProduct
from astropy.time import Time
from airflow.exceptions import AirflowSkipException
from peewee import fn
from typing import Optional
from sdssdb.peewee.sdss5db import catalogdb

from astra.sdss.operators.base import SDSSOperator

try:
    catalogdb.database.set_profile(config["sdss5_database"]["profile"])
except (KeyError, TypeError):
    None

try:
    catalogdb.SDSS_DR17_APOGEE_Allvisits
except AttributeError:

    # TODO: this table not yet in sdssdb, will be in next version
    #       remove this after sdssdb 0.5.4
    class SDSS_DR17_APOGEE_Allvisits(catalogdb.CatalogdbModel):

        visit_id = catalogdb.TextField(primary_key=True)

        class Meta:
            table_name = "sdss_dr17_apogee_allvisits"

    Visit = SDSS_DR17_APOGEE_Allvisits

else:
    Visit = catalogdb.SDSS_DR17_APOGEE_Allvisits
    print(f"ANDY: remove SDSS_DR17_APOGEE_Allvisits in {__file__}")

Star = catalogdb.SDSS_DR17_APOGEE_Allstarmerge


class ApStarOperator(SDSSOperator):

    """An operator to retrieve and ingest ApStar data products created in SDSS-IV."""

    release = "dr17"
    filetype = "apVisit"

    def __init__(self, return_id_kind: Optional[str] = "data_product", **kwargs):
        super(SDSSOperator, self).__init__(**kwargs)
        self._available_return_id_kinds = ["data_product", "catalog"]
        if return_id_kind not in self._available_return_id_kinds:
            raise ValueError(
                f"return_id_kind must be one of {self._available_return_id_kinds}"
            )
        self.return_id_kind = return_id_kind
        return None

    def execute(self, context: dict):
        """
        Execute this operator.

        :param context:
            The Airflow context dictionary.
        """
        ds = context.get("ds", None)
        prev_ds = context.get("prev_ds", None)

        # I first thought I'd have to query Star, matched on Visit, but actually the Star merge table
        # munges columns like 'telescope' -> 'telescopes' (all visits) and ignores things like field.
        # That means the Star merge table is nearly useless for our purposes here (we only use it to
        # get gaia_source_id and cross-match to the catalog).

        # When catalog_to_sdss_dr17_apogee_allvisits exists, we won't need to cross-match via Stars at all
        fields = (Visit.apogee_id, Visit.telescope, Visit.field, Star.gaia_source_id)
        q = (
            Visit.select(*fields)
            .distinct()
            .join(Star, on=(Star.apogee_id == Visit.apogee_id))
        )

        if not (ds is None and prev_ds is None):
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
            path = self.path_instance.full(self.filetype, **kwds)
            if not os.path.exists(path):
                errors.append(
                    {"detail": path, "origin": star, "reason": "File does not exist"}
                )
                log.warning(
                    f"Error ingesting path {path} from {star}: {errors[-1]['reason']}"
                )
                continue
            else:
                # Cross-match to catalogdb.
                (catalogid,) = (
                    catalogdb.Catalog.select(catalogdb.Catalog.catalogid)
                    .join(catalogdb.CatalogToTIC_v8)
                    .join(catalogdb.TIC_v8)
                    .join(catalogdb.Gaia_DR3)
                    .where(catalogdb.Gaia_DR3.source_id == star.gaia_source_id)
                    .tuples()
                    .first()
                )
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
        if len(ids) == 0:
            raise AirflowSkipException(f"No data products ingested.")

        index = self._available_return_id_kinds.index(self.return_id_kind)
        return [each[index] for each in ids]


class ApVisitOperator(SDSSOperator):

    """An operator to retrieve and ingest ApVisit data products created in SDSS-IV."""

    release = "dr17"
    filetype = "apVisit"

    def __init__(self, return_id_kind: Optional[str] = "data_product", **kwargs):
        super(SDSSOperator, self).__init__(**kwargs)
        self._available_return_id_kinds = ["data_product", "catalog"]
        if return_id_kind not in self._available_return_id_kinds:
            raise ValueError(
                f"return_id_kind must be one of {self._available_return_id_kinds}"
            )
        self.return_id_kind = return_id_kind
        return None

    def execute(self, context: dict):
        """
        Execute this operator.

        :param context:
            The Airflow context dictionary.
        """
        ds = context.get("ds", None)
        prev_ds = context.get("prev_ds", None)

        fields = (
            Visit.apogee_id,
            Visit.mjd,
            Visit.plate,
            Visit.telescope,
            Visit.fiberid,
            Visit.field,
        )
        if not (ds is None and prev_ds is None):
            # Doing a direct query between Visit and Star causes a sequential scan across Star, which is really
            # slow. Instead let's try a sub-query.
            sq = (
                Visit.select(*fields)
                .where(Visit.mjd.between(Time(prev_ds).mjd, Time(ds).mjd))
                .alias("sq")
            )
            q = (
                Star.select(
                    Star.gaia_source_id,
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
            q = (
                Visit.select(Star.gaia_source_id, *fields)
                .join(Star, on=(Star.apogee_id == Visit.apogee_id))
                .objects()
            )

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
                    {"detail": path, "origin": visit, "reason": "File does not exist"}
                )
                log.warning(
                    f"Error ingesting path {path} from {visit}: {errors[-1]['reason']}"
                )
                continue
            else:
                # Cross-match to catalogdb.
                (catalogid,) = (
                    catalogdb.Catalog.select(catalogdb.Catalog.catalogid)
                    .join(catalogdb.CatalogToTIC_v8)
                    .join(catalogdb.TIC_v8)
                    .join(catalogdb.Gaia_DR3)
                    .where(catalogdb.Gaia_DR3.source_id == visit.gaia_source_id)
                    .tuples()
                    .first()
                )
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
        if len(ids) == 0:
            raise AirflowSkipException(f"No data products ingested.")

        index = self._available_return_id_kinds.index(self.return_id_kind)
        return [each[index] for each in ids]
