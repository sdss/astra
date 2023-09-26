from peewee import fn, chunked
from tqdm import tqdm
from astra.models.source import Source

def migrate_carton_assignments_to_bigbitfield(
    where=(fn.length(Source.carton_flags) == 0),
    batch_size=500,
    limit=None
):
    from astra.migrations.sdss5db.targetdb import Target, CartonToTarget, Carton

    raise RuntimeError

    # Retrieve sources which have gaia identifiers but not astrometry
    q = (
        Source
        .select()
        .where(where)
        .order_by(Source.sdss5_catalogid_v1.asc())
        .limit(limit)
        .iterator()
    )

    updated = 0
    total = limit or Source.select().where(where).count()
    
    with tqdm(total=total) as pb:
        for batch in chunked(q, batch_size):          
            sources = { ea.sdss5_catalogid_v1: ea for ea in batch }
            q_cartons = (
                CartonToTarget
                .select(
                    Target.catalogid,
                    CartonToTarget.carton_pk,
                )
                .join(Target)
                .where(Target.catalogid.in_(list(sources.keys())))
                .tuples()
                .iterator()
            )

            update = []
            for sdss5_catalogid_v1, carton_pk in q_cartons:
                print(sdss5_catalogid_v1, carton_pk)
                s = sources[sdss5_catalogid_v1]
                s.carton_flags.set_bit(carton_pk)
                update.append(s)
            
            
            if update:
                updated += (
                    Source
                    .bulk_update(
                        list(set(update)),
                        fields=[
                            Source.carton_flags
                        ]
                    )
                )
            
            pb.update(updated)

    return updated