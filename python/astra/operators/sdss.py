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

@lru_cache
def lookup_keys(release, filetype):
    return SDSSPath(release=release).lookup_keys(filetype)


def get_or_create_data_product_from_apogee_drpdb(
        origin, 
        release=None,
        filetype=None,
        with_source=True, 
        with_metadata=True, 
        headers=None,
    ):
    """
    Get or create a data product entry in the astra database from an apogee_drpdb origin.
    
    :param release: [optional]
        Supply a release. Otherwise this will be read from `origin.release`.

    :param filetype: [optional]
        The filetype of the data product. If `None`, this will be read from `origin.filetype`.

    :param with_source: [optional]
        Get or create a Source origin for this data product.
    
    :param with_metadata: [optional]
        Get or create a Metadata origin for this data product.

    :param headers: [optional]
        A dictionary of headers to read in and store with the metadata. This should be a 
        dictionary with metadata names as keys, and the values should be a two length
        tuple that contains the HDU index (0-index) and the header key. 
    """

    release = release or origin.release
    filetype = filetype or origin.filetype

    keys = lookup_keys(release, filetype)
    data_product, created = DataProduct.get_or_create(
        release=release,
        filetype=filetype,
        kwargs={ k: getattr(origin, k) for k in keys }
    )
    error = {
        "detail": data_product.path,
        "origin": origin,
    }

    # Check the file exists.
    if not os.path.exists(data_product.path):
        error.update(reason="File does not exist")
        data_product.delete_instance()
        return (False, error)
    
    result = dict(data_product=data_product)
    if not with_source:
        return (True, result)
        
    # Load the file.
    try:
        image = fits.open(data_product.path)
    except:
        log.exception(f"Could not open path {data_product.path}")
        error.update(reason="Could not open file")
        data_product.delete_instance()
        return (False, error)

    # Read headers.
    header_details = OrderedDict([("catalogid", (0, "CATID"))])
    header_details.update(headers or {})

    metadata = {}
    for key, (hdu, header) in header_details.items():
        try:
            metadata[key] = image[hdu].header[header]
        except:
            reason = f"Could not read header {header} in HDU {hdu}"
            log.exception(f"{reason} of {data_product.path}")
            error.update(reason=reason)
            data_product.delete_instance()
            return (False, error)
    
    catalogid = metadata.pop("catalogid")
    if catalogid <= 0:
        log.exception(f"CatalogID for {data_product} {data_product.path} is {catalogid}")
        error.update(reason="CatalogID is not positive")
        data_product.delete_instance()
        return (False, error)

    source, _ = Source.get_or_create(catalogid=catalogid)
    SourceDataProduct.get_or_create(
        source=source,
        data_product=data_product
    )
    result.update(source=source)
    if with_metadata:
        data_product.metadata = metadata
        with database.atomic() as tx:
            DataProduct.update(metadata=metadata).where(DataProduct.pk == data_product.pk).execute()
        #data_product.update(metadata=metadata).execute()

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




class BaseApogeeDRPOperator(BaseOperator):

    def get_or_create_data_products(self, prev_ds, ds, model, description, where=None, **kwargs):
    
        log.info(f"Looking for {description} products created between {prev_ds} and {ds}")
        q = (
            model.select()
                 .where(model.created.between(prev_ds, ds))
        )
        if where is not None:
            q = q.where(where)

        N = q.count()
        if N == 0:
            raise AirflowSkipException(f"No {description} products created between {prev_ds} and {ds}")

        pks = []
        errors = []
        for origin in q:
            success, result = get_or_create_data_product_from_apogee_drpdb(
                origin,
                **kwargs
            )
            if success:
                log.info("Data product {data_product} matched to {source}".format(**result))
                pks.append(result["data_product"].pk)
            else:
                log.warning("{reason} ({detail}) from {origin}".format(**result))
                errors.append(result)

        log.info(f"Found {N} rows.")
        log.info(f"Created {len(pks)} data products.")
        log.info(f"Encountered {len(errors)} errors.")
        if len(errors) == N:
            raise AirflowFailException(f"{N}/{N} data products had errors")
        return pks



class ApVisitOperator(BaseApogeeDRPOperator):

    def execute(self, context):
        return self.get_or_create_data_products(
            context["prev_ds"],
            context["ds"],
            Visit,
            "SDSS-V ApVisit",
            where=(Visit.catalogid > 0),
            headers={
                "snr": (0, "SNR"),
                "naxis1": (1, "NAXIS1")
            }
        )




class ApStarOperator(BaseApogeeDRPOperator):
    """
    Generate data model products in the Astra database for all new ApStar
    files produced since the operator was last executed.

    :param releases: [optional]
        The relevant SDSS data releases. If `None` is given then this will be inferred based on
        the execution date.
    """

    ui_color = "#ffb09c"

    def execute(self, context):
        return self.get_or_create_data_products(
            context["prev_ds"],
            context["ds"],
            Star,
            "SDSS-V ApStar",
            where=(Star.ngoodvisits > 0) & (Star.catalogid > 0),
            headers={
                "snr": (0, "SNR"),
            }
        )


