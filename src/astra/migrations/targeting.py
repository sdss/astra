import datetime
import re
from importlib import resources
from peewee import fn, chunked
from tqdm import tqdm
from astra.models.base import database
from astra.models.source import Source
from astra.utils import log, expand_path
from astropy.table import Table
from astropy.table import join
from astra.migrations.utils import NoQueue

def get_carton_to_bit_mapping():
    """Read the carton-to-bit mapping from the highest-versioned file shipped with sdss-semaphore."""
    etc_dir = resources.files("sdss_semaphore").joinpath("etc")
    pattern = re.compile(r"sdss5_target_(\d+)_with_groups\.csv$")
    candidates = [
        (int(match.group(1)), entry)
        for entry in etc_dir.iterdir()
        if (match := pattern.match(entry.name))
    ]
    _, path = max(candidates, key=lambda x: x[0])
    return Table.read(path)

def migrate_targeting_cartons(where=(Source.sdss5_target_flags == b""), batch_size=500, queue=None):

    from astra.migrations.sdss5db.targetdb import Target, CartonToTarget
    from astra.migrations.sdss5db.catalogdb import CatalogdbModel

    if queue is None:
        queue = NoQueue()

    bit_mapping = {}
    for row in get_carton_to_bit_mapping():
        row_as_dict = dict(zip(row.keys(), row.values()))
        bit_mapping[row_as_dict["carton_pk"]] = row_as_dict

    class SDSS_ID_Flat(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_flat"

    q = (
        Source
        .select()
        .where(where)
    )

    queue.put(dict(total=None, description="Migrating targeting cartons"))
    for chunk in chunked(q.iterator(), batch_size):
        update_dict = {}
        chunk_dict = { s.sdss_id: s for s in chunk }

        q_cartons = (
            SDSS_ID_Flat
            .select(
                SDSS_ID_Flat.sdss_id,
                CartonToTarget.carton_pk,
            )
            .join(Source, on=(SDSS_ID_Flat.sdss_id == Source.sdss_id))
            .switch(SDSS_ID_Flat)
            .join(Target, on=(SDSS_ID_Flat.catalogid == Target.catalogid))
            .join(CartonToTarget, on=(Target.pk == CartonToTarget.target_pk))
            .where(
                    SDSS_ID_Flat.sdss_id.in_(list(chunk_dict.keys()))
                &   (SDSS_ID_Flat.rank == 1)
            )
            .tuples()
        )
        now = datetime.datetime.now()
        for sdss_id, carton_pk in q_cartons.iterator():
            try:
                bit = bit_mapping[carton_pk]["bit"]
            except KeyError:
                None # todo
            else:
                chunk_dict[sdss_id].sdss5_target_flags.set_bit(bit)
                chunk_dict[sdss_id].modified = now
                update_dict[sdss_id] = chunk_dict[sdss_id]

        if update_dict:
            with database.atomic():
                (
                    Source
                    .bulk_update(
                        update_dict.values(),
                        fields=[Source.sdss5_target_flags, Source.modified]
                    )
                )
        queue.put(dict(advance=batch_size))

    queue.put(Ellipsis)
