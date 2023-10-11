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
    from_cache=False
):

    bit_mapping = {}
    for row in get_carton_to_bit_mapping():
        row_as_dict = dict(zip(row.keys(), row.values()))
        bit_mapping[row_as_dict["carton_pk"]] = row_as_dict

    # Retrieve sources which have gaia identifiers but not astrometry
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
    
    if from_cache:
        import pickle
        with open(expand_path("~/20230926.pkl"), "rb") as fp:
            q_cartons = pickle.load(fp)

    else:
        from astra.migrations.sdss5db.targetdb import Target, CartonToTarget, Assignment
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
            .join(Target, on=(SDSS_ID_Flat.catalogid == Target.catalogid))
            .join(CartonToTarget, on=(Target.pk == CartonToTarget.target_pk))
            .join(Assignment, on=(Assignment.carton_to_target_pk == CartonToTarget.pk))
            .tuples()
            .iterator()
        )

    missing, update, sources = (set(), {}, {})
    for source in tqdm(q, desc="Preparing sources", total=1):
        sources[source.sdss_id] = source

    n_skipped, n_applied = (0, 0)
    for sdss_id, carton_pk in tqdm(q_cartons, total=len(q_cartons) if from_cache else 1, desc="Assiging cartons to sources"):
        try:
            s = sources[sdss_id]
        except:
            missing.add(sdss_id)
            n_skipped += 1
        else:
            s.sdss5_target_flags.set_bit(bit_mapping[carton_pk]["bit"])
            update[sdss_id] = s
            n_applied += 1
    
    log.info(f"Flagged {n_applied} source-carton assignments, and skipped {n_skipped} ({len(missing)} sources missing)")

    if len(missing) > 0:
        log.warning(f"There were {len(missing)} sdss_ids with target assignments that are not in Astra's database")

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

    return updated