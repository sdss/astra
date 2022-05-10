import numpy as np
import os
from astropy.table import Table, MaskedColumn

from peewee import ForeignKeyField, JOIN, Expression, Alias, Field
from astra import log, config, __version__
from astra.database.astradb import Source, Task, DataProduct, Output, TaskInputDataProducts, SourceDataProduct, TaskOutput

# TODO: refactor our own catalogdb to use this one.
from sdssdb.peewee.sdss5db.catalogdb import database as catalogdb_database
catalogdb_database.set_profile("astra")
from sdssdb.peewee.sdss5db.catalogdb import TIC_v8, Catalog, Gaia_DR2, CatalogToTIC_v8, TwoMassPSC


def export_table(model, path=None):
    
    if path is None:
        path = os.path.expandvars(f"$MWM_ASTRA/{__version__}/{model._meta.name}-all.fits")

    # Ignore any `meta` columns for now
    fields = list(filter(
        lambda c: not isinstance(c, ForeignKeyField) and c.name != "meta", 
        model._meta.sorted_fields
    ))


    q_results = (
        Task.select(
                Source.catalogid,
                DataProduct.release,
                DataProduct.filetype,
                DataProduct.kwargs,
                Task.id.alias("task_id"),
                Task.version,
                Task.time_total,
                Task.created,
                Task.parameters,
                Output.id.alias("output_id"),
                *fields
            )
            .join(TaskInputDataProducts)
            .join(DataProduct)
            .join(SourceDataProduct)
            .join(Source)
            .switch(Task)
            .join(TaskOutput, JOIN.LEFT_OUTER)
            .join(Output)
            .join(model)
            .order_by(Task.id.asc(), Output.id.asc())
            .dicts()
    )

    log.debug(f"Querying {q_results}")

    results = []
    catalogids = set()
    parameter_sets = []
    last_kwargs = None
    for N, row in enumerate(q_results, start=1):
        catalogids.add(row["catalogid"])
        if "parameters" in row:
            parameters = row.pop("parameters")
            try:
                parameter_sets.append(frozenset(parameters))
            except:
                log.warning(f"Failed to serialize parameters: {parameters}")
            
        row["created"] = row["created"].isoformat()
        last_kwargs = row.pop("kwargs")
        row.update(last_kwargs)
    
        results.append(row)

    log.info(f"We have {N} result rows")
    log.info(f"Querying for metadata on {len(catalogids)} sources")

    # Supply with metadata from the catalog
    q_meta = (
        Catalog.select(
                    Catalog.catalogid,
                    Catalog.ra,
                    Catalog.dec,
                    TIC_v8.id.alias("tic_v8_id"),
                    Gaia_DR2.source_id.alias("gaia_dr2_source_id"),
                    Gaia_DR2.parallax,
                    Gaia_DR2.phot_bp_mean_mag,
                    Gaia_DR2.phot_rp_mean_mag,
                    TwoMassPSC.j_m,
                    TwoMassPSC.h_m,
                    TwoMassPSC.k_m,
            )
            .distinct(Catalog.catalogid)
            .join(CatalogToTIC_v8, JOIN.LEFT_OUTER)
            .join(TIC_v8)
            .join(Gaia_DR2, JOIN.LEFT_OUTER)
            .switch(TIC_v8)
            .join(TwoMassPSC, JOIN.LEFT_OUTER)
            .where(Catalog.catalogid.in_(list(catalogids)))
            .order_by(Catalog.catalogid.asc())
            .dicts()
    )

    meta = { row["catalogid"]: row for row in q_meta}
    missing_metadata = []

    names = []
    ignore_names = ("parameters", )
    for query in (q_meta, q_results):
        for field in query._returning:
            if isinstance(field, Expression):
                name = field.rhs
            elif isinstance(field, Field):
                name = field.name
            elif isinstance(field, Alias):
                name = field._alias
            else:
                raise RuntimeError(f"Cannot get name for field type ({type(field)} ({field}) of {query}")

            if name == "kwargs":
                for name in last_kwargs.keys():
                    names.append(name)
            else:
                if name not in names and name not in ignore_names:
                    names.append(name)

    M, C = (0, 0)
    for row in results:
        catalogid = row["catalogid"]
        try:
            row.update(meta[catalogid])
        except KeyError:
            if catalogid not in missing_metadata:
                log.warning(f"No metadata found for catalogid {catalogid}!")
                missing_metadata.append(catalogid)
            M += 1
        else:
            C += 1

    if len(missing_metadata) > 0:
        log.warning(f"In total there are {len(missing_metadata)} catalog sources without metadata!")

    # If NONE of the rows have metadata, then we will get an error when we try to build a table.
    # We should fill the first result with empty values.
    if C == 0:
        missing_field_names = set(names).difference(results[0])
        results[0].update({ name: np.nan for name in missing_field_names })
        log.warning(f"ALL sources are missing metadata!")

    table = Table(
        data=results,
        names=names,            
    )
    # Fix dtypes etc.
    fill_values = {
        float: np.nan,
        int: -1
    }
    for index, (name, dtype) in enumerate(table.dtype.descr):
        if dtype == "|O":
            # Objects.
            mask = (table[name] == None)
            if all(mask) or (not any(mask) and len(set(table[name])) == 1):
                # All Nones, probably. Delete.
                del table[name]
            else:
                data = np.array(table[name])
                del table[name]

                kwds = dict(name=name)
                if any(mask):
                    dtype = type(data[~mask][0])
                    fill_value = fill_values[dtype]
                    data[mask] = fill_value
                    kwds.update(
                        mask=mask,
                        dtype=dtype,
                        fill_value=fill_value
                    )

                table.add_column(
                    MaskedColumn(data, **kwds),
                    index=index
                )

    table.write(path, overwrite=True)

    log.info(f"Created file: {path}")
    return True 

    """
    data_product, _ = DataProduct.get_or_create(
        release="sdss5",
        filetype="full",
        kwargs=dict(full=path)
    )

    log.info(f"Created data product {data_product} at {path}")

    return data_product.id
    """