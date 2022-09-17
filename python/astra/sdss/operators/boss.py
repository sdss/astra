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
        require_mwm_carton: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.release = release
        self.filetype = filetype
        self.run2d = run2d
        self.require_mwm_carton = require_mwm_carton
        return None

    def execute(self, context):

        data = Table.read(
            expand_path(f"$BOSS_SPECTRO_REDUX/{self.run2d}/spAll-{self.run2d}.fits")
        )

        prev_ds, ds = (context.get("prev_ds", None), context.get("ds", None))
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

        if self.require_mwm_carton:
            log.info(f"Requiring a MWM carton")
            from astra.sdss.operators.mwm import STELLAR_CARTONS
            from astra.database.targetdb import Target, CartonToTarget, Carton

            sq_carton = (
                Carton.select(Carton.pk)
                .where(Carton.carton.in_(STELLAR_CARTONS))
                .alias("sq_carton")
            )

            done = {}
            indices = np.where(mask)[0]
            count = len(indices)
            log.info(f"This could take up to {count/(60 * 200):.0f} minutes")

            for index, catalogid in zip(indices, data["CATALOGID"][mask]):
                try:
                    mask[index] = done[catalogid]
                except KeyError:
                    q = (
                        Target.select(Target.pk)
                        .join(CartonToTarget)
                        .join(
                            sq_carton, on=(CartonToTarget.carton_pk == sq_carton.c.pk)
                        )
                        .where(Target.catalogid == catalogid)
                    )
                    mask[index] = done[catalogid] = q.exists()
            log.info(f"Found {np.sum(mask)} products with MWM cartons (of {count})")

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
