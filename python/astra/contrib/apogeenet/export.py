
import os
from astropy.table import Table
from collections import OrderedDict
from sqlalchemy import func
from tqdm import tqdm

from astra.utils import log
from astra.database import (astradb, apogee_drpdb, catalogdb, session)

def export_to_table(output_path, overwrite=True):
    """
    Export the APOGEENet database results to a table.

    :param output_path:
        The disk location where to write the table to.
    
    :param overwrite: [optional]
        Overwrite any
    """

    output_path = os.path.expandvars(os.path.expanduser(output_path))
    if not overwrite and os.path.exists(output_path):
        raise OSError(f"path '{output_path}' already exists and asked not to overwrite it")

    sq = session.query(
            astradb.ApogeeNet.output_pk.label("output_pk"),
            func.json_object_agg(
                astradb.Parameter.parameter_name,
                astradb.Parameter.parameter_value
            ).label("parameters")
        )\
        .filter(astradb.ApogeeNet.output_pk == astradb.TaskInstance.output_pk)\
        .filter(astradb.TaskInstance.pk == astradb.TaskInstanceParameter.ti_pk)\
        .filter(astradb.TaskInstanceParameter.parameter_pk == astradb.Parameter.pk)\
        .group_by(astradb.ApogeeNet)\
        .subquery(with_labels=True)

    q = session.query(
            astradb.TaskInstance,
            astradb.ApogeeNet, 
            func.cardinality(astradb.ApogeeNet.snr),
            sq.c.parameters
        )\
        .filter(sq.c.output_pk == astradb.ApogeeNet.output_pk)\
        .filter(sq.c.output_pk == astradb.TaskInstance.output_pk)

    total, = session.query(func.sum(func.cardinality(astradb.ApogeeNet.snr))).first()

    table_columns = OrderedDict([
        ("ti_pk", []),
        ("run_id", []),
        ("release", []),
        ("apred", []),
        ("field", []),
        ("healpix", []),
        ("telescope", []),
        ("obj", []),
        ("spectrum_index", []),
    ])
    column_names = ("snr", "teff", "u_teff", "logg", "u_logg", "fe_h", "u_fe_h", "bitmask_flag")
    for cn in column_names:
        table_columns[cn] = []

    with tqdm(total=total, unit="spectra") as pb:
    
        for task_instance, result, N, parameters in q.yield_per(1):
            for i in range(N):
                table_columns["ti_pk"].append(result.ti_pk)
                table_columns["run_id"].append(task_instance.run_id)
                table_columns["release"].append(parameters["release"])
                table_columns["apred"].append(parameters["apred"])
                table_columns["field"].append(parameters.get("field", ""))
                table_columns["healpix"].append(parameters.get("healpix", ""))
                table_columns["telescope"].append(parameters["telescope"])
                table_columns["obj"].append(parameters["obj"])
                table_columns["spectrum_index"].append(i)

                for column_name in column_names:
                    table_columns[column_name].append(getattr(result, column_name)[i])
                
                pb.update(1)
    
    log.info(f"Creating table with {total} rows")
    table = Table(data=table_columns)
    log.info(f"Table created.")

    log.info(f"Writing to {output_path}")
    table.write(output_path, overwrite=overwrite)
    log.info("Done")

    return table_columns