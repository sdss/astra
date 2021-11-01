
import numpy as np
from astropy.io import fits
from sdss_access import SDSSPath
from sqlalchemy.sql import exists
from sqlalchemy import func
from tqdm import tqdm
from collections import OrderedDict
from astropy.table import Table

from astra.database import (astradb, session as astra_session)
from astra.database.sdssdb import (apogee_drpdb, catalogdb, session as sdssdb_session)
from astra.utils import (log, flatten)


def create_source_table(database_model_name, output_path, format=None, overwrite=False, filter_by_kwargs=None, limit=None):
    """
    Create a table of sources and their results from the database. 
    
    If there are mutiple results per source (e.g., from individual visits), then only the first result is included.

    Parameters
    ----------
    database_model_name : str
        Name of the database model to query.
    output_path : str
        Path to the output file.
    format : str
        Format of the output file.
    filter_by_kwargs : dict
        Keyword arguments to pass to the database query.
    limit : int
        Limit the number of results.
    """

    rows = _get_results(
        database_model_name, 
        spectrum_index=0, 
        filter_by_kwargs=filter_by_kwargs, 
        limit=limit
    )

    table = Table(rows=rows)
    table.write(output_path, format=format, overwrite=overwrite)
    log.info(f"Wrote source table with {len(table)} rows to {output_path}")
    return None


def create_visit_table(database_model_name, output_path, format=None, overwrite=False, filter_by_kwargs=None, limit=None):
    """
    Create a table of visits and their results from the database. 
    
    Parameters
    ----------
    database_model_name : str
        Name of the database model to query.
    output_path : str
        Path to the output file.
    format : str
        Format of the output file.
    filter_by_kwargs : dict
        Keyword arguments to pass to the database query.
    limit : int
        Limit the number of results.
    """

    rows = _get_results(
        database_model_name, 
        spectrum_index=None, 
        filter_by_kwargs=filter_by_kwargs, 
        limit=limit
    )

    table = Table(rows=rows)
    table.write(output_path, format=format, overwrite=overwrite)
    log.info(f"Wrote visit table with {len(table)} rows to {output_path}")
    return None

    
def _count_rows(q):
    count_q = q.statement.with_only_columns([func.count()]).order_by(None)
    return q.session.execute(count_q).scalar()


def _get_results(database_model_name, spectrum_index=None, filter_by_kwargs=None, limit=None):

    # Get the database model
    database_model = getattr(astradb, database_model_name)
    if filter_by_kwargs is None:
        filter_by_kwargs = dict()

    q = astra_session.query(
            astradb.TaskInstance, 
            func.json_object_agg(
                astradb.Parameter.parameter_name, 
                astradb.Parameter.parameter_value
            ),
            astradb.TaskInstanceMeta,
            database_model,
        ).filter(
            astradb.TaskInstance.output_pk == database_model.output_pk,
            astradb.TaskInstance.pk == astradb.TaskInstanceMeta.ti_pk,
            astradb.TaskInstance.pk == astradb.TaskInstanceParameter.ti_pk,
            astradb.TaskInstanceParameter.parameter_pk == astradb.Parameter.pk,
        ).filter_by(
            **filter_by_kwargs
        ).group_by(astradb.TaskInstance, astradb.TaskInstanceMeta, database_model)

    if limit is not None:
        q = q.limit(limit)

    rows = []
    for ti, parameters, meta, result in tqdm(q.yield_per(1), total=q.count()):
        
        row = OrderedDict([
            # Source information.
            ("catalogid", meta.catalogid),
            ("ra", meta.ra or np.nan),
            ("dec", meta.dec or np.nan),
            ("pmra", meta.pmra or np.nan),
            ("pmdec", meta.pmdec or np.nan),
            ("parallax", meta.parallax or np.nan),
            ("gaia_dr2_source_id", meta.gaia_dr2_source_id or -1),
            # Task information.
            ("ti_pk", ti.pk),
            # Parameters (minimal)
            ("release", parameters.get("release", "")),
            ("obj", parameters.get("obj", "")),
            ("healpix", parameters.get("healpix", -1)),
            ("telescope", parameters.get("telescope", "")),
        ])

        # Add the result information.
        ignore_keys = ("output_pk", "ti_pk", "associated_ti_pks")
        if spectrum_index is None:
            # Get all results.
            N = len(result.snr)
            for i in range(N):
                this_row = row.copy()
                this_row["spectrum_index"] = i
                for key in result.__table__.columns.keys():
                    if key in ignore_keys or key.startswith("_"): continue

                    value = getattr(result, key)
                    if isinstance(value, (tuple, list)):
                        value = value[i]
                    this_row[key] = value or np.nan
                
                rows.append(this_row)
    
        else:
            # Only single result.
            for key in result.__table__.columns.keys():
                if key in ignore_keys or key.startswith("_"): continue
            
                value = getattr(result, key)

                    
                if isinstance(value, (tuple, list)):
                    value = value[spectrum_index]
            
                row[key] = value or np.nan

            rows.append(row)
        
    return rows