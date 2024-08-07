from peewee import fn, chunked
from tqdm import tqdm
from astra.models.source import Source
from astra.utils import log, expand_path
from astropy.table import Table
from astropy.table import join

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



def migrate_carton_assignments_to_bigbitfield(
    where=None,
    batch_size=500,
    limit=None,
    full_output=False
):

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

    updated = 0
    total = limit or Source.select().where(where).count()
    
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
        .iterator()
    )

    missing, update, sources = (set(), {}, {})
    for source in tqdm(q, desc="Preparing sources", total=1):
        sources[source.sdss_id] = source

    n_skipped, n_applied, not_marked = (0, 0, {})
    for sdss_id, carton_pk in tqdm(q_cartons, total=1, desc="Assiging cartons to sources"):
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
                    log.warning(f"No bit mapping for carton pk={carton_pk} and sdss_id={sdss_id} (source_pk={s.pk})")
                not_marked[carton_pk].append(sdss_id)
                n_skipped += 1
            else:
                s.sdss5_target_flags.set_bit(bit)
                update[sdss_id] = s
                n_applied += 1        

    log.info(f"Flagged {n_applied} source-carton assignments, and skipped {n_skipped} ({len(missing)} sources missing)")
    
    if len(not_marked) > 0:
        log.warning(f"There were {len(not_marked)} cartons that we did not update because the bit was not in the semaphore file")
        for carton_pk, sdss_ids in not_marked.items():
            log.warning(f"  Carton pk={carton_pk} had {len(sdss_ids)} (e.g., {sdss_ids[0]}) and no bit exists in the semaphore file")

    if len(missing) > 0:
        log.warning(f"There were {len(missing)} sdss_ids with target assignments that are not in Astra's database (e.g., {missing[0]})")

    updated = 0
    with tqdm(desc="Updating", total=len(update)) as pb:

        for chunk in chunked(update.values(), batch_size):
            updated += (
                Source
                .bulk_update(
                    chunk,
                    fields=[
                        Source.sdss5_target_flags
                    ]
                )
            )
            pb.update(batch_size)

    if full_output:
        return (updated, missing, not_marked)
    return updated