import os
from typing import Optional
from airflow.models.baseoperator import BaseOperator
from airflow.exceptions import AirflowSkipException
from sdss_access import SDSSPath
from astra.database.astradb import database, DataProduct, Source, SourceDataProduct
from astra import log
from astra.utils import expand_path
from astropy.time import Time
import numpy as np

from astropy.table import Table

from functools import lru_cache


@lru_cache
def path_instance(release):
    return SDSSPath(release=release)


@lru_cache
def lookup_keys(release, filetype):
    return path_instance(release).lookup_keys(filetype)


class BossSpectrumOperator(BaseOperator):
    """
    A base operator for working with SDSS-V BOSS spectrum data products.

    This operator will generate task instances based on BOSS reduced data products it finds that were
    *observed* in the operator execution period.
    """

    ui_color = "#A0B9D9"

    def __init__(
        self,
        *,
        release: Optional[str] = None,
        filetype: Optional[str] = "specFull",
        run2d: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.release = release
        self.filetype = filetype
        self.run2d = run2d
        return None

    def execute(self, context):

        data = Table.read(
            expand_path(f"$BOSS_SPECTRO_REDUX/{self.run2d}/spAll-{self.run2d}.fits")
        )

        prev_ds, ds = (context["prev_ds"], context["ds"])
        if prev_ds is None:
            # Schedule once only
            mask = np.ones(len(data), dtype=bool)
        else:
            mjd_start, mjd_end = list(map(lambda x: Time(x).mjd, (prev_ds, ds)))
            mask = (mjd_end > data["MJD"]) & (data["MJD"] >= mjd_start)

        N = sum(mask)
        if N == 0:
            raise AirflowSkipException(
                f"No products for {self} observed between {prev_ds} and {ds}"
            )

        errors = []
        ids = []
        for row in data[mask]:
            catalogid = row["CATALOGID"]
            # From Sean Morrison:
            # Note if you have looked at the v604 outputs we used a p at the end of the plate ID like 15000p,
            # this was to uncertainty about if the plateids were going to be unique to the fps, but in v609
            # this will change since the ids are unique fieldid<16000 is plates, > is fps.
            isplate = "p" if row["FIELD"] < 16_000 else ""
            kwds = dict(
                fieldid=f"{row['FIELD']:0>6.0f}",
                mjd=int(row["MJD"]),
                catalogid=int(catalogid),
                run2d=row["RUN2D"],
                isplate=isplate,
            )
            path = path_instance(self.release).full(self.filetype, **kwds)
            if not os.path.exists(path):
                error = {"detail": path, "origin": row, "reason": "File does not exist"}
                errors.append(error)
                continue

            # We need the ZWARNING metadata downstream.
            with database.atomic() as txn:
                data_product, _ = DataProduct.get_or_create(
                    release=self.release,
                    filetype=self.filetype,
                    kwargs=kwds,
                )
                source, _ = Source.get_or_create(catalogid=catalogid)
                SourceDataProduct.get_or_create(
                    source=source, data_product=data_product
                )

            log.info(f"Data product {data_product} matched to {source}")
            ids.append(data_product.id)

        N = sum(mask)
        log.info(f"Found {N} rows.")
        log.info(f"Created {len(ids)} data products.")
        if len(errors):
            log.warning(f"Encountered {len(errors)} errors:")
            for error in errors:
                log.warning(f"{error['reason']}: {error['detail']}")

        if len(errors) == N:
            raise AirflowSkipException(f"{N}/{N} data products had errors")
        return ids
