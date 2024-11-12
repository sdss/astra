from peewee import fn, chunked
from tqdm import tqdm
from astra.models.base import database
from astra.models.source import Source
from astra.utils import log, expand_path
from astropy.table import Table
from astropy.table import join
from astra.migrations.utils import NoQueue

def get_carton_to_bit_mapping():
    return Table.read(expand_path("$MWM_ASTRA/aux/targeting-bits/sdss5_target_1_with_groups.csv"))


def merge_carton_to_bit_mapping_and_meta():
    
    targets = Table.read(expand_path("$MWM_ASTRA/aux/targeting-bits/sdss5_target_1.csv"))
    meta = Table.read(expand_path("$MWM_ASTRA/aux/targeting-bits/sdss5_target_1_groups.csv"))

    # assign the metadata to each bit mapping
    bit_mapping = join(targets, meta, keys=("label", ))
    
    # drop unnecessary columns (for us) and rename others
    del bit_mapping["pk"]

    bit_mapping.rename_column("program_1", "program")
    bit_mapping.rename_column("program_2", "alt_program")
    bit_mapping.rename_column("altname", "alt_name")
    bit_mapping.sort(["bit"])
    bit_mapping.rename_column("Carton_pk", "carton_pk")
    
    bit_mapping.write(expand_path("$MWM_ASTRA/aux/targeting-bits/sdss5_target_1_with_groups.csv"))

def migrate_targeting_cartons(where=(Source.sdss5_target_flags == b""), batch_size=1000, queue=None):

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

    queue.put(dict(total=q.count()))
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
            .where(SDSS_ID_Flat.sdss_id.in_(list(chunk_dict.keys())))
            .tuples()
        )
        for sdss_id, carton_pk in q_cartons.iterator():
            try:
                bit = bit_mapping[carton_pk]["bit"]
            except KeyError:
                None # todo
            else:
                chunk_dict[sdss_id].sdss5_target_flags.set_bit(bit)
                update_dict[sdss_id] = chunk_dict[sdss_id]

        if update_dict:
            with database.atomic():
                (
                    Source
                    .bulk_update(
                        update_dict.values(),
                        fields=[Source.sdss5_target_flags]
                    )
                )
        queue.put(dict(advance=batch_size))
    
    queue.put(Ellipsis)

        



def migrate_carton_assignments_to_bigbitfield(
    where=None,
    batch_size=500,
    limit=None,
    full_output=False,
    queue=None,
):
    if queue is None:
        queue = NoQueue()

    bit_mapping = {}
    for row in get_carton_to_bit_mapping():
        row_as_dict = dict(zip(row.keys(), row.values()))
        bit_mapping[row_as_dict["carton_pk"]] = row_as_dict

    q = (
        Source
        .select()
    )
    if where:
        q = q.where(where)
    
    q = (
        q
        .limit(limit)
        .iterator()
    )

    missing, update, sources = (set(), {}, {})    
    for source in q:
        sources[source.sdss_id] = source
    
    from astra.migrations.sdss5db.targetdb import Target, CartonToTarget
    from astra.migrations.sdss5db.catalogdb import CatalogdbModel

    class SDSS_ID_Flat(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_flat"

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
        .tuples()
    )

    warnings = []
    n_skipped, n_applied, not_marked = (0, 0, {})
    queue.put(dict(description="Assigning cartons to sources", total=q_cartons.count(), completed=0))
    for sdss_id, carton_pk in q_cartons.iterator():
        try:
            s = sources[sdss_id]
        except:
            missing.add(sdss_id)
            n_skipped += 1
        else:
            try:
                bit = bit_mapping[carton_pk]["bit"]
            except KeyError:
                if carton_pk not in not_marked:
                    not_marked.setdefault(carton_pk, [])
                    warnings.append(f"No bit mapping for carton pk={carton_pk} and sdss_id={sdss_id} (source_pk={s.pk})")
                not_marked[carton_pk].append(sdss_id)
                n_skipped += 1
            else:
                s.sdss5_target_flags.set_bit(bit)
                update[sdss_id] = s
                n_applied += 1        
        
        queue.put(dict(advance=1))

    #log.info(f"Flagged {n_applied} source-carton assignments, and skipped {n_skipped} ({len(missing)} sources missing)")
    
    if len(not_marked) > 0:
        warnings.append(f"There were {len(not_marked)} cartons that we did not update because the bit was not in the semaphore file")
        for carton_pk, sdss_ids in not_marked.items():
            warnings.append(f"  Carton pk={carton_pk} had {len(sdss_ids)} (e.g., {sdss_ids[0]}) and no bit exists in the semaphore file")

    if len(missing) > 0:
        warnings.append(f"There were {len(missing)} sdss_ids with target assignments that are not in Astra's database (e.g., {missing[0]})")

    queue.put(dict(description="Updating sources with targeting bits", total=len(update), completed=0))
    updated = 0
    for chunk in chunked(update.values(), batch_size):
        with database.atomic():                
            updated += (
                Source
                .bulk_update(
                    chunk,
                    fields=[
                        Source.sdss5_target_flags
                    ]
                )
            )
        queue.put(dict(advance=batch_size))

    queue.put(Ellipsis)
    if full_output:
        return (updated, missing, not_marked)
    return updated