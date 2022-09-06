import os
from astra.database.astradb import database, DataProduct, Source, SourceDataProduct

assert os.environ["ASTRA_DATABASE_URL"] is not None


from astra.database.apogee_drpdb import Star

from astra.database.astradb import (
    database,
    Task,
    TaskInputDataProducts,
    DataProduct,
    Output,
    TaskOutput,
    Source,
    SourceDataProduct,
    create_tables,
)

create_tables()

from astra.contrib.apogeenet.database import ApogeeNet

if not ApogeeNet.table_exists():
    ApogeeNet.create_table()


# Create some fake sources, data products.
import numpy as np

np.random.seed(0)

N = 1000

from sdssdb.peewee.sdss5db.catalogdb import database as catalogdb_database

catalogdb_database.set_profile("astra")
from sdssdb.peewee.sdss5db.catalogdb import (
    TIC_v8,
    Catalog,
    Gaia_DR2,
    CatalogToTIC_v8,
    TwoMassPSC,
)

catalogids = [ea.catalogid for ea in Catalog.select().limit(N)]

tasks = []
with database.atomic() as tx:
    for catalogid in catalogids:
        source = Source.create(catalogid=catalogid)
        data_product = DataProduct.create(
            release="sdss5", filetype="full", kwargs={"full": f"{catalogid}.fits"}
        )
        SourceDataProduct.create(source=source, data_product=data_product)

        task = Task.create(name="test", parameters={"this": "that"}, version="test")
        TaskInputDataProducts.create(task=task, data_product=data_product)
        tasks.append(task)


# Create tasks.
V = np.random.randint(1, 50, size=N)  # visits
V[
    (V == 2) | (V == 3)
] = 4  # there's either 1 visit, or at least 4 (2 x visits + 1 stack)

with database.atomic() as tx:
    for task, v in zip(tasks, V):
        for _ in range(v):
            output = Output.create()
            TaskOutput.create(task=task, output=output)
            ApogeeNet.create(
                task=task,
                output=output,
                snr=np.random.uniform(1, 100),
                teff=np.random.uniform(4000, 6000),
                logg=np.random.uniform(0, 5),
                fe_h=np.random.uniform(-2, 0),
                u_teff=np.random.uniform(1, 1000),
                u_logg=np.random.uniform(1, 5),
                u_fe_h=np.random.uniform(1, 5),
                teff_sample_median=1,
                logg_sample_median=1,
                fe_h_sample_median=1,
                bitmask_flag=1,
            )

# Now do the big query.
from peewee import JOIN, fn

# For retrieving all results.
q_results = (
    Task.select(
        Source.catalogid,
        DataProduct.release,
        DataProduct.filetype,
        DataProduct.kwargs["full"].alias("full"),
        Task.pk.alias("task_pk"),
        Task.version,
        Task.time_total,
        Task.created,
        Task.parameters,
        Output.pk.alias("output_pk"),
        ApogeeNet.snr,
        ApogeeNet.teff,
        ApogeeNet.logg,
        ApogeeNet.snr,
        ApogeeNet.teff,
        ApogeeNet.logg,
        ApogeeNet.fe_h,
        ApogeeNet.u_teff,
        ApogeeNet.u_logg,
        ApogeeNet.u_fe_h,
        ApogeeNet.teff_sample_median,
        ApogeeNet.logg_sample_median,
        ApogeeNet.fe_h_sample_median,
        ApogeeNet.bitmask_flag,
    )
    .join(TaskInputDataProducts)
    .join(DataProduct)
    .join(SourceDataProduct)
    .join(Source)
    .switch(Task)
    .join(TaskOutput, JOIN.LEFT_OUTER)
    .join(Output)
    .join(ApogeeNet)
    .group_by(ApogeeNet)
    .order_by(Task.pk.asc(), Output.pk.asc())
    .dicts()
)

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
    .where(Catalog.catalogid.in_(catalogids))
    .order_by(Catalog.catalogid.asc())
    .dicts()
)

meta = {row["catalogid"]: row for row in q_meta}


results = []
parameter_sets = []
for row in q_results:
    try:
        row.update(meta[row["catalogid"]])
    except:
        print(f"No metadata at all for catalogid {row['catalogid']}")
        raise
    parameter_sets.append(frozenset(row.pop("parameters").items()))

    row["created"] = row["created"].isoformat()
    results.append(row)


meta = dict()
N_parameter_sets = len(set(parameter_sets))
if N_parameter_sets == 1:
    meta.update(parameter_sets[0])
else:
    # We should include these as columns.
    raise a

from astropy.table import Table, MaskedColumn

from peewee import Field, Alias, Expression

names = []
ignore_names = ("parameters",)
for query in (q_meta, q_results):
    for column in query._returning:
        if isinstance(column, Expression):
            name = column.rhs
        elif isinstance(column, Field):
            name = column.name
        elif isinstance(column, Alias):
            name = column._alias
        else:
            raise WhatDo

        if name not in names and name not in ignore_names:
            names.append(name)

t = Table(data=results, names=names, meta=meta)

# Fix dtypes etc.
fill_values = {float: np.nan, int: -1}
for index, (name, dtype) in enumerate(t.dtype.descr):
    if dtype == "|O":
        # Objects.
        if len(set(t[name])) == 1:
            # All Nones, probably. Delete.
            del t[name]
        else:
            mask = t[name] == None
            data = np.array(t[name])
            del t[name]

            kwds = dict(name=name)
            if any(mask):
                dtype = type(data[~mask][0])
                fill_value = fill_values[dtype]
                data[mask] = fill_value
                kwds.update(mask=mask, dtype=dtype, fill_value=fill_value)

            t.add_column(MaskedColumn(data, **kwds), index=index)

# Done!
