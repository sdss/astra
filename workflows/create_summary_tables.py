

# Create summary files.
import numpy as np
from astra.database import (schema, session)
from astropy.table import Table

tables = [
    ("apogeenet", schema.astra.apogeenet),
    ("thepayne", schema.astra.thepayne_apstar),
    ("classify-apstar", schema.astra.classify_apstar),
    ("classify-apvisit", schema.astra.classify_apvisit),
]

for (suffix, table) in tables:
    
    names = [column.name for column in table.__table__.columns]

    row = session.query(table).first()
    array_columns = [cn for cn in names if isinstance(getattr(row, cn), list)]


    data = []
    for row in session.query(table).all():

        if len(array_columns):
            cardinality = len(getattr(row, array_columns[0]))
        else:
            cardinality = 1
        for j in range(cardinality):
            values = []
            for cn in names:
                value = getattr(row, cn)
                if cn in array_columns:
                    value = value[j]
                
                #if value is None:
                #    value = ""
                values.append(value)

            # Index.
            values.append(j)

            data.append(tuple(values))

    names.append("index")
    
    summary_table = Table(rows=data, names=names)
    
    # Delete unnecessary columns.
    del summary_table["modified"]
    if not len(array_columns):
        del summary_table["index"]

    path = f"/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/astra/0.1.11/summary-{suffix}.fits"
    summary_table.write(path, overwrite=True)

    print(f"Created {path} with {len(summary_table)} rows")
        

# The FERRE ones are proxies.
suffix = "ferre-apstar"
proxy_table = schema.astra.ferre_apstar
table = schema.astra.ferre

names = [c.name for c in proxy_table.__table__.columns]
for c in table.__table__.columns:
    if c.name not in names:
        names.append(c.name)

data = []
array_names = [
'teff',
'logg',
'metals',
'o_mg_si_s_ca_ti',
'n',
'c',
'log10vdop',
'lgvsini',
'u_teff',
'u_logg',
'u_metals',
'u_o_mg_si_s_ca_ti',
'u_n',
'u_c',
'u_log10vdop',
'u_lgvsini',
'log_snr_sq',
'log_chisq_fit'
]

from tqdm import tqdm
for proxy_row in tqdm(session.query(proxy_table).all()):

    # Resolve the proxy.
    row = session.query(table).filter(table.task_id==proxy_row.proxy_task_id).first()

    data_row = row.__dict__.copy()
    data_row.update(proxy_row.__dict__)    

    cardinality = len(data_row["teff"])
    
    for i in range(cardinality):
        values = []
        for name in names:
            value = data_row[name]
            if name in array_names:
                if value is None:
                    value = np.nan
                else:
                    value = value[i]
            
            values.append(value)

        values.append(i)
        data.append(values)

names.append("index")
summary_table = Table(rows=data, names=names)

single_values = {}
for name in names:
    unique_values = set(summary_table[name])
    if len(unique_values) == 1:
        unique_value = list(unique_values)[0]
        if isinstance(unique_value, (str, bytes)) and len(unique_value) > 10:
            single_values[name] = list(unique_values)[0]
            del summary_table[name]

remove_names = [
    "input_lsf_path", "ferre_kwds", "modified", "frozen_parameters"
]
for name in remove_names:
    del summary_table[name]

path = f"/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/astra/0.1.11/summary-{suffix}.fits"
summary_table.write(path, overwrite=True)

print(f"Created {path} with {len(summary_table)} rows")
