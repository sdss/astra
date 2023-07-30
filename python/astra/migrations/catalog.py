
from typing import Optional
from tqdm import tqdm
from peewee import chunked, IntegerField
from astra.models.source import Source
from astra.migrations.sdss5db.utils import get_approximate_rows
from astra.utils import log, flatten



def migrate_healpix(
    where=(
            Source.healpix.is_null()
        &   Source.ra.is_null(False)
        &   Source.dec.is_null(False)
    ),
    batch_size: Optional[int] = 500,
    limit: Optional[int] = None,
    nside: Optional[int] = 128,
    lonlat: Optional[bool] = True
):
    """
    Migrate HEALPix values for any sources that have positions, but no HEALPix assignment.

    :param batch_size: [optional]
        The batch size to use when upserting data.    
    
    :param limit: [optional]
        Limit the initial catalog queries for testing purposes.
    
    :param nside: [optional]
        The number of sides to use for the HEALPix map.
    
    :param lonlat: [optional]
        The HEALPix map is oriented in longitude and latitude coordinates.
    """
    from healpy import ang2pix
    
    log.info(f"Migrating healpix")
    q = (
        Source
        .select(
            Source.id,
            Source.ra,
            Source.dec,
        )
        .where(where)
        .limit(limit)
    )    
    
    updated, total = (0, limit or q.count())
    with tqdm(total=total) as pb:
        for batch in chunked(q.iterator(), batch_size):
            for record in batch:
                record.healpix = ang2pix(nside, record.ra, record.dec, lonlat=lonlat)
            updated += (
                Source
                .bulk_update(
                    batch,
                    fields=[Source.healpix],
                )
            )
            pb.update(len(batch))

    log.info(f"Updated {updated} records")
    return updated



def migrate_sources_from_sdss5_catalogdb(batch_size: Optional[int] = 500, limit: Optional[int] = None):
    """
    Migrate all catalog sources stored in the SDSS-V database.

    This creates a unique identifier per astronomical source (akin to a `sdss_id`) and links all possible
    catalog identifiers (`catalogdb.catalog.catalogid`) to those unique sources.
            
    :param batch_size: [optional]
        The batch size to use when upserting data.    
    
    :param limit: [optional]
        Limit the initial catalog queries for testing purposes.
    
    :returns:
        A tuple of new `sdss_id` identifiers created.
    """
    
    from astra.migrations.sdss5db.catalogdb import CatalogdbModel
    
    class Catalog_ver25_to_ver31_full_unique(CatalogdbModel):

        id = IntegerField(primary_key=True)

        class Meta:
            table_name = 'catalog_ver25_to_ver31_full_unique'

    log.info(f"Querying catalogdb.catalog_ver25_to_ver31_unique")

    q = (
        Catalog_ver25_to_ver31_full_unique
        .select(
            Catalog_ver25_to_ver31_full_unique.id,
            Catalog_ver25_to_ver31_full_unique.lowest_catalogid,
            Catalog_ver25_to_ver31_full_unique.highest_catalogid,
        )
        .limit(limit)
        .tuples()
        .iterator()
    )

    # Sometimes the highest_catalogid appears twice. There's a good reason for this.
    # I just don't know what it is. But we need unique-ness, and we need to link to
    # the lower catalog identifiers.
    next_sdss_id = 1
    source_data, lookup_sdss_id_from_catalog_id, lookup_catalog_id_from_sdss_id = ({}, {}, {})
    for sdss_id, lowest, highest in tqdm(q, total=limit or get_approximate_rows(Catalog_ver25_to_ver31_full_unique)):

        # Do we already have an sdss_id assigned to this highest catalog identifier?
        sdss_id_1 = lookup_sdss_id_from_catalog_id.get(highest, None)
        sdss_id_2 = lookup_sdss_id_from_catalog_id.get(lowest, None)

        if sdss_id_1 is not None and sdss_id_2 is not None and sdss_id_1 != sdss_id_2:
            # We need to amalgamate these two.
            affected = []
            affected.extend(lookup_catalog_id_from_sdss_id[sdss_id_1])
            affected.extend(lookup_catalog_id_from_sdss_id[sdss_id_2])
            
            # merge both into sdss_id_1
            source_data[sdss_id_1] = dict(
                sdss_id=sdss_id_1,
                sdss5_catalogid_v1=max(affected)
            )
            for catalogid in affected:
                lookup_sdss_id_from_catalog_id[catalogid] = sdss_id_1

            lookup_catalog_id_from_sdss_id[sdss_id_1] = affected
            
            del source_data[sdss_id_2]
            del lookup_catalog_id_from_sdss_id[sdss_id_2]
        
        else:
            sdss_id = sdss_id_1 or sdss_id_2
            if sdss_id is None:
                sdss_id = 0 + next_sdss_id
                next_sdss_id += 1
        
            lookup_catalog_id_from_sdss_id.setdefault(sdss_id, [])
            lookup_catalog_id_from_sdss_id[sdss_id].extend((lowest, highest))

            lookup_sdss_id_from_catalog_id[lowest] = sdss_id
            lookup_sdss_id_from_catalog_id[highest] = sdss_id
            source_data[sdss_id] = dict(
                sdss_id=sdss_id,
                sdss5_catalogid_v1=highest
            )

    log.info(f"There are {len(source_data)} unique `sdss_id` entries so far")

    class Catalog_ver25_to_ver31_full_all(CatalogdbModel):

        id = IntegerField(primary_key=True)

        class Meta:
            table_name = 'catalog_ver25_to_ver31_full_all'
    
    log.info(f"Querying catalogdb.catalog_ver25_to_ver31_full_all")

    q = (
        Catalog_ver25_to_ver31_full_all
        .select(
            Catalog_ver25_to_ver31_full_all.lowest_catalogid,
            Catalog_ver25_to_ver31_full_all.highest_catalogid
        )
        .limit(limit)
        .tuples()
        .iterator()
    )
    
    for lowest, highest in tqdm(q, total=limit or get_approximate_rows(Catalog_ver25_to_ver31_full_all)):

        sdss_id_1 = lookup_sdss_id_from_catalog_id.get(highest, None)
        sdss_id_2 = lookup_sdss_id_from_catalog_id.get(lowest, None)

        if sdss_id_1 is not None and sdss_id_2 is not None and sdss_id_1 != sdss_id_2:
            # We need to amalgamate these two.
            affected = []
            affected.extend(lookup_catalog_id_from_sdss_id[sdss_id_1])
            affected.extend(lookup_catalog_id_from_sdss_id[sdss_id_2])
            
            # merge both into sdss_id_1
            source_data[sdss_id_1] = dict(
                sdss_id=sdss_id_1,
                sdss5_catalogid_v1=max(affected)
            )
            for catalogid in affected:
                lookup_sdss_id_from_catalog_id[catalogid] = sdss_id_1

            lookup_catalog_id_from_sdss_id[sdss_id_1] = affected
            
            del source_data[sdss_id_2]
            del lookup_catalog_id_from_sdss_id[sdss_id_2]
        
        else:
            sdss_id = sdss_id_1 or sdss_id_2
            if sdss_id is None:
                sdss_id = 0 + next_sdss_id
                next_sdss_id += 1
        
            lookup_catalog_id_from_sdss_id.setdefault(sdss_id, [])
            lookup_catalog_id_from_sdss_id[sdss_id].extend((lowest, highest))

            lookup_sdss_id_from_catalog_id[lowest] = sdss_id
            lookup_sdss_id_from_catalog_id[highest] = sdss_id
            source_data[sdss_id] = dict(
                sdss_id=sdss_id,
                sdss5_catalogid_v1=highest
            )
            
    log.info(f"There are now {len(source_data)} unique `sdss_id` entries so far")

    # Create the Source
    new_source_ids = []
    with database.atomic():
        # Need to chunk this to avoid SQLite limits.
        with tqdm(desc="Upserting", unit="sources", total=len(source_data)) as pb:
            for chunk in chunked(source_data.values(), batch_size):
                new_source_ids.extend(
                    Source
                    .insert_many(chunk)
                    .on_conflict_ignore()
                    .returning(Source.sdss_id)
                    .tuples()
                    .execute()
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()

    log.info(f"Inserted {len(new_source_ids)} new sources")

    log.info("Linking catalog identifiers to SDSS identifiers")
    
    data_generator = (
        dict(catalogid=catalogid, sdss_id=sdss_id) 
        for catalogid, sdss_id in lookup_sdss_id_from_catalog_id.items()
    )
    
    with database.atomic():
        with tqdm(desc="Linking catalog identifiers to unique sources", total=len(lookup_sdss_id_from_catalog_id)) as pb:
            for chunk in chunked(data_generator, batch_size):
                (
                    SDSSCatalog
                    .insert_many(chunk)
                    .on_conflict_ignore()
                    .returning(SDSSCatalog.catalogid)
                    .tuples()
                    .execute()
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()
    
    return tuple(new_source_ids)





def migrate_catalog_sky_positions(batch_size=1000):

    from astra.migrations.sdss5db.catalogdb import Catalog

    log.info(f"Querying for sources without basic metadata")
    where_source = (
            Source.ra.is_null()
        &   (Source.sdss5_catalogid_v1.is_null(False))        
    )
    q = (
        Source
        .select(
            Source.sdss5_catalogid_v1
        )
        .where(where_source)
        .tuples()
        .iterator()
    )

    log.info(f"Querying catalogdb.catalog for basic catalog metadata")
    catalog_data = {}
    n_q = 0
    for batch in tqdm(chunked(q, batch_size), total=1):        
        q_catalog = (
            Catalog
            .select(
                Catalog.catalogid.alias("sdss5_catalogid_v1"),
                Catalog.ra,
                Catalog.dec,
                Catalog.lead,
                Catalog.version_id
            )
            .where(Catalog.catalogid.in_(flatten(batch)))
            .tuples()
            .iterator()
        )

        for sdss5_catalogid_v1, ra, dec, lead, version_id in q_catalog:
            catalog_data[sdss5_catalogid_v1] = dict(
                ra=ra,
                dec=dec,
                version_id=version_id,
                lead=lead
            )
        n_q += len(batch)

    q = (
        Source
        .select()
        .where(where_source)
    )

    updated = 0
    with tqdm(total=n_q) as pb:
        for batch in chunked(q.iterator(), batch_size):
            for source in batch:
                try:
                    for key, value in catalog_data[source.sdss5_catalogid_v1].items():
                        setattr(source, key, value)
                except:
                    continue
            updated += (
                Source
                .bulk_update(
                    batch,
                    fields=[
                        Source.ra,
                        Source.dec,
                        Source.version_id,
                        Source.lead
                    ],
                )
            )
            pb.update(len(batch))            
            

    # You might think that we should only do subsequent photometry/astrometry queries for things that were 
    # actually targeted, but you'd be wrong. We need to include everything from SDSS-IV / DR17 too, which
    # was not assigned to any SDSS-V carton.

    '''    
    # Only do the carton/target cleverness if we are using a postgresql database.
    if isinstance(database, PostgresqlDatabase):
        log.info(f"Querying cartons and target assignments")    

        from astra.migrations.sdss5db.targetdb import Target, CartonToTarget

        q = (
            CartonToTarget
            .select(
                Target.catalogid,
                CartonToTarget.carton_pk
            )
            .join(Target, on=(CartonToTarget.target_pk == Target.pk))
            .tuples()
            .iterator()
        )
        lookup_cartons_by_catalogid = {}
        for catalogid, carton_pk in tqdm(q, total=get_approximate_rows(CartonToTarget)):
            lookup_cartons_by_catalogid.setdefault(catalogid, [])
            lookup_cartons_by_catalogid[catalogid] = carton_pk
        
        # We will actually assign these to sdss_id entries later on, when it's time to create
        # the sources.
        raise NotImplementedError
    
    else:
        log.warning(f"Not including carton and target assignments right now")
    '''
    log.info(f"Inserting sources")

        


def migrate_cata():

    log.info("Preparing data for catalog queries..")

    log.info(f"Getting Gaia DR3 information")

    # Add Gaia DR3 identifiers
    
    q = (
        Gaia_DR3
        .select(
            CatalogToGaia_DR3.catalogid,
            Gaia_DR3.source_id.alias("gaia_dr3_source_id"),
            Gaia_DR3.parallax.alias("plx"),
            Gaia_DR3.parallax_error.alias("e_plx"),
            Gaia_DR3.pmra,
            Gaia_DR3.pmra_error.alias("e_pmra"),
            Gaia_DR3.pmdec.alias("pmde"),
            Gaia_DR3.pmdec_error.alias("e_pmde"),
            Gaia_DR3.phot_g_mean_mag.alias("g_mag"),
            Gaia_DR3.phot_bp_mean_mag.alias("bp_mag"),
            Gaia_DR3.phot_rp_mean_mag.alias("rp_mag"),
        )
        .join(CatalogToGaia_DR3, on=(CatalogToGaia_DR3.target_id == CatalogToGaia_DR3.source_id))
        .where(
            CatalogToGaia_DR3.catalogid.in_(reference_catalogids)
        )
        .dicts()
    )
    for row in tqdm(q):
        sdss_id = lookup_sdss_id_from_catalog_id[row["catalogid"]]
        source_data[sdss_id].update(**row)

    log.info(f"Querying TIC v8")
    
    q = (
        CatalogToTIC_v8
        .select(
            CatalogToTIC_v8.catalogid,
            CatalogToTIC_v8.target_id
        )
        .where(
            CatalogToTIC_v8.catalogid.in_(reference_catalogids)
        )
        .dicts()
    )
    for catalogid, tic_v8_id in q:
        sdss_id = lookup_sdss_id_from_catalog_id[catalogid]
        source_data[sdss_id].update(tic_v8_id=tic_v8_id)
    
    # Now do SDSSCatalog
    
    raise a

    '''
    log.info(f"Upserting sources..")

    new_sources = []
    with database.atomic():
        # Need to chunk this to avoid SQLite limits.
        with tqdm(desc="Upserting", unit="sources", total=len(sources)) as pb:
            for chunk in chunked(sources, batch_size):
                new_sources.extend(
                    Source
                    .insert_many(chunk)
                    .on_conflict_ignore()
                    .returning(Source.sdss_id, Source.sdss5_catalogid_v1)
                    .tuples()
                    .execute()
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()

    raise a
    sdss_ids = dict(new_sources)
    sources_to_catalogids = [{ "sdss_id": sdss_id, "catalogid": catalogid } for sdss_id, catalogid in sources_to_catalogids]

    log.info("Upserting links")

    with database.atomic():
        with tqdm(desc="Linking catalog identifiers to unique sources", total=len(sources_to_catalogids)) as pb:
            for chunk in chunked(sources_to_catalogids, batch_size):
                (
                    SDSSCatalog
                    .insert_many(chunk)
                    .on_conflict_ignore()
                    .tuples()
                    .execute()
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()
    
    log.info(f"Inserting basic catalog information")
    new_catalogids = [c for s, c in new_sources]
    '''


    # 
    cids = {}
    for s, c in new_sources:
        cids.setdefault(c, [])
        cids[c].append(s)

    data = []
    for sdss5_catalogid_v1, ra, dec, lead, version_id in q:
        for sdss_id in cids[sdss5_catalogid_v1]:
            data.append(
                {
                    "ra": ra,
                    "dec": dec,
                    "lead": lead,
                    "version_id": version_id,
                    "sdss_id": sdss_id
                }
            )

    with database.atomic():
        with tqdm(desc="Upserting basic catalog information", total=len(data)) as pb:
            for chunk in chunked(data, batch_size):
                (
                    Source
                    .insert_many(chunk)
                    .on_conflict_replace()
                    .execute()
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()        

    # Add Gaia DR3 identifiers
    log.info(f"Querying Gaia DR3")
    
    q = (
        CatalogToGaia_DR3
        .select(
            CatalogToGaia_DR3.catalog,
            CatalogToGaia_DR3.target
        )
        .where(
            CatalogToGaia_DR3.catalog.in_(new_catalogids)
        )
        .tuples()
    )
    data = []
    for catalogid, targetid in q:
        for sdss_id in cids[catalogid]:
            data.append({
                "sdss_id": sdss_id,
                "gaia_dr3_source_id": targetid
            })

    with database.atomic():
        with tqdm(desc="Linking Gaia DR3", total=len(data)) as pb:
            for chunk in chunked(data, batch_size):
                (
                    Source
                    .insert_many(chunk)
                    .on_conflict_replace()
                    .execute()
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()               

    # Add Gaia DR2 identifiers

    # Add TIC v8 identifier
    # Add Gaia DR3 identifiers
    log.info(f"Querying TIC v8")
    
    q = (
        CatalogToTIC_v8
        .select(
            CatalogToTIC_v8.catalogid,
            CatalogToTIC_v8.target_id
        )
        .where(
            CatalogToTIC_v8.catalogid.in_(new_catalogids)
        )
        .tuples()
    )
    data = []
    for catalogid, targetid in q:
        for sdss_id in cids[catalogid]:
            data.append({
                "sdss_id": sdss_id,
                "tic_v8_id": targetid
            })

    with database.atomic():
        with tqdm(desc="Linking TIC v8", total=len(data)) as pb:
            for chunk in chunked(data, batch_size):
                (
                    Source
                    .insert_many(chunk)
                    .on_conflict_replace()
                    .execute()
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()               

    raise a
    raise a



if __name__ == "__main__":
    from astra.models.source import Source
    models = [Spectrum, ApogeeVisitSpectrum, Source, SDSSCatalog]
    #database.drop_tables(models)
    if models[0].table_exists():
        database.drop_tables(models)
    database.create_tables(models)

    #migrate_apvisit_from_sdss5_apogee_drpdb()

    migrate_sources_from_sdss5_catalogdb()

