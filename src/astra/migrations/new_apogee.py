import concurrent.futures
import subprocess
import numpy as np
from peewee import JOIN, chunked, Case, fn, SQL, EXCLUDED, IntegrityError
from typing import Optional

from astra.migrations.utils import enumerate_new_spectrum_pks, upsert_many, NoQueue
from astra.utils import expand_path, flatten, log
from tqdm import tqdm

def migrate_apogee_spectra_from_sdss5_apogee_drpdb(apred: str, max_mjd: Optional[int] = None, queue=None, limit=None, incremental=True, **kwargs):
    queue = queue or NoQueue()

    # Let's do visits first.
    v_n_new_sources, v_n_new_spectra, v_n_updated_spectra = migrate_apogee_visits(apred, max_mjd=max_mjd, queue=queue, limit=limit, incremental=incremental, **kwargs)
    
    # Now do co-added spectra.
    c_n_new_sources, c_n_new_spectra, c_n_updated_spectra = migrate_apogee_coadds(apred, max_mjd=max_mjd, queue=queue, limit=limit, incremental=incremental, **kwargs)
    
    # Now we need to check all the apStar files for the ApogeeVisitSpectrumInApStar entries,
    # and for whether it was dithered or not.
    new_apstar_visit_spectra, failed_to_match = migrate_apogee_visits_in_apStar_files(apred, max_workers=16, queue=queue, limit=limit, batch_size=1000)
    
    queue.put(Ellipsis)
    log.info(f"APOGEE {apred}: {v_n_new_spectra} new visits; {v_n_updated_spectra} updated visits; {v_n_new_sources} new sources")
    log.info(f"APOGEE {apred}: {c_n_new_spectra} new coadds; {c_n_updated_spectra} updated coadds; {c_n_new_sources} new sources")
    
    return None



def migrate_apogee_visits_in_apStar_files(apred: str, max_workers=16, queue=None, limit=None, batch_size=1000):

    from astra.models.base import database
    from astra.models.apogee import ApogeeCoaddedSpectrumInApStar, ApogeeVisitSpectrumInApStar, ApogeeVisitSpectrum

    queue = queue or NoQueue()

    executor = concurrent.futures.ProcessPoolExecutor(max_workers)
    q = (
        ApogeeCoaddedSpectrumInApStar
        .select()
        .where(
            (ApogeeCoaddedSpectrumInApStar.apred == apred)
        &   (ApogeeCoaddedSpectrumInApStar.mean_fiber.is_null())
        )
        .limit(limit)
        .iterator()
    )

    apStar_spectra, futures = ({}, [])
    total = 0
    queue.put(dict(description="Getting apStar metadata", total=total, completed=0))
    #print("Getting apStar metadata")
    for total, spectrum in enumerate(q, start=1):
        futures.append(executor.submit(_get_apstar_metadata, spectrum))
        apStar_spectra[spectrum.spectrum_pk] = spectrum
        queue.put(dict(advance=1))


    visit_spectrum_data = []
    failed_spectrum_pks = []
    queue.put(dict(description="Collecting apStar metadata", total=total, completed=0))
    #print("Collecting apStar metadata")
    for future in concurrent.futures.as_completed(futures):
        result = future.result()         
        for spectrum_pk, metadata in result.items():                
            if metadata is None:
                failed_spectrum_pks.append(spectrum_pk)
                continue
                            
            spectrum = apStar_spectra[spectrum_pk]

            mjds = []
            sfiles = [metadata[f"SFILE{i}"] for i in range(1, int(metadata["NVISITS"]) + 1)]
            for sfile in sfiles:
                #if spectrum.telescope == "apo1m":
                #    #"$SAS_BASE_DIR/dr17/apogee/spectro/redux/{apred}/visit/{telescope}/{field}/{mjd}/apVisit-{apred}-{mjd}-{reduction}.fits"
                #    # sometimes it is stored as a float AHGGGGHGGGGHGHGHGH
                #    mjds.append(int(float(sfile.split("-")[2])))
                #else:
                #    mjds.append(int(float(sfile.split("-")[3])))
                #    # "$SAS_BASE_DIR/dr17/apogee/spectro/redux/{apred}/visit/{telescope}/{field}/{plate}/{mjd}/{prefix}Visit-{apred}-{plate}-{mjd}-{fiber:0>3}.fits"
                # NOTE: For SDSS5 data this is index 4: 'apVisit-1.2-apo25m-5339-59715-103.fits'
                mjds.append(int(float(sfile.split("-")[4])))                    

            assert len(sfiles) == int(metadata["NVISITS"])
            
            spectrum.snr = float(metadata["SNR"])
            spectrum.mean_fiber = float(metadata["MEANFIB"])
            spectrum.std_fiber = float(metadata["SIGFIB"])
            spectrum.n_good_visits = int(metadata["NVISITS"])
            spectrum.n_good_rvs = int(metadata["NVISITS"])
            spectrum.v_rad = float(metadata.get("VRAD", metadata.get("VHBARY")))
            spectrum.e_v_rad = float(metadata["VERR"])
            spectrum.std_v_rad = float(metadata["VSCATTER"])
            spectrum.median_e_v_rad = float(metadata.get("VERR_MED", np.nan))
            spectrum.spectrum_flags = metadata["STARFLAG"]

            # The MJDS in the apStar file only list the MJDs that were included in the stack. 
            # But there could be other MJDs which were not included in the stack.
            # TODO: To be consistent elsewhere we should probably not update these based on 
            spectrum.min_mjd = min(mjds)
            spectrum.max_mjd = max(mjds)

            star_kwds = dict(
                source_pk=spectrum.source_pk,
                release=spectrum.release,
                filetype=spectrum.filetype,
                apred=spectrum.apred,
                apstar=spectrum.apstar,
                obj=spectrum.obj,
                telescope=spectrum.telescope,
                #field=spectrum.field,
                #prefix=spectrum.prefix,
                #reduction=spectrum.obj if spectrum.telescope == "apo1m" else None           
            )
            for i, sfile in enumerate(sfiles, start=1):
                #if spectrum.telescope != "apo1m":
                #    plate = sfile.split("-")[2]
                #else:
                #    # plate not known..
                #    plate = metadata["FIELD"].strip()
                mjd = int(sfile.split("-")[4])
                plate = sfile.split("-")[3]

                kwds = star_kwds.copy()
                kwds.update(
                    mjd=mjd,
                    fiber=int(metadata[f"FIBER{i}"]),
                    plate=plate
                )
                visit_spectrum_data.append(kwds)

            queue.put(dict(advance=1))            

    queue.put(dict(description="Updating apStar metadata", total=total, completed=0))
    #print(f"Updating apStar metadata for {total} spectra")
    for chunk in chunked(apStar_spectra.values(), batch_size):
        queue.put(dict(advance=(
            ApogeeCoaddedSpectrumInApStar  
            .bulk_update(
                chunk,
                fields=[
                    ApogeeCoaddedSpectrumInApStar.snr,
                    ApogeeCoaddedSpectrumInApStar.mean_fiber,
                    ApogeeCoaddedSpectrumInApStar.std_fiber,
                    ApogeeCoaddedSpectrumInApStar.n_good_visits,
                    ApogeeCoaddedSpectrumInApStar.n_good_rvs,
                    ApogeeCoaddedSpectrumInApStar.v_rad,
                    ApogeeCoaddedSpectrumInApStar.e_v_rad,
                    ApogeeCoaddedSpectrumInApStar.std_v_rad,
                    ApogeeCoaddedSpectrumInApStar.median_e_v_rad,
                    ApogeeCoaddedSpectrumInApStar.spectrum_flags,
                    ApogeeCoaddedSpectrumInApStar.min_mjd,
                    ApogeeCoaddedSpectrumInApStar.max_mjd
                ]
            )
        )))

    q = (
        ApogeeVisitSpectrum
        .select(
            ApogeeVisitSpectrum.obj, # using this instead of source_pk because some apogee_ids have two different sources
            ApogeeVisitSpectrum.spectrum_pk,
            ApogeeVisitSpectrum.telescope,
            ApogeeVisitSpectrum.plate,
            ApogeeVisitSpectrum.mjd,
            ApogeeVisitSpectrum.fiber
        )
        .where(ApogeeVisitSpectrum.apred == apred)
        .tuples()
    )
    queue.put(dict(description="Matching to ApogeeVisitSpectrum", total=q.count(), completed=0))
    #print(f"Matching to ApogeeVisitSpectrum for {q.count()} spectra")

    drp_spectrum_data = {}
    for obj, spectrum_pk, telescope, plate, mjd, fiber in q:
        drp_spectrum_data.setdefault(obj, {})
        key = "_".join(map(str, (telescope, plate, mjd, fiber)))
        drp_spectrum_data[obj][key] = spectrum_pk
        queue.put(dict(advance=1))


    queue.put(dict(description="Linking to ApogeeVisitSpectrum", total=len(visit_spectrum_data), completed=0))
    #print(f"Linking to ApogeeVisitSpectrum for {len(visit_spectrum_data)} spectra")
    only_ingest_visits = []
    failed_to_match_to_drp_spectrum_pk = []
    for spectrum_pk, visit in enumerate_new_spectrum_pks(visit_spectrum_data):
        key = "_".join(map(str, [visit[k] for k in ("telescope", "plate", "mjd", "fiber")]))
        try:
            drp_spectrum_pk = drp_spectrum_data[visit["obj"]][key]
        except:
            failed_to_match_to_drp_spectrum_pk.append((spectrum_pk, visit))
        else:            
            visit.update(
                spectrum_pk=spectrum_pk,
                drp_spectrum_pk=drp_spectrum_pk
            )
            only_ingest_visits.append(visit)
        queue.put(dict(advance=1))

    if len(failed_to_match_to_drp_spectrum_pk) > 0:
        log.warning(f"There were {len(failed_to_match_to_drp_spectrum_pk)} spectra that we could not match to DRP spectra")
        log.warning(f"Example: {failed_to_match_to_drp_spectrum_pk[0]}")

    queue.put(dict(description="Upserting ApogeeVisitSpectrumInApStar spectra", total=len(only_ingest_visits), completed=0))
    #print(f"Upserting ApogeeVisitSpectrumInApStar spectra for {len(only_ingest_visits)} spectra")
    n_apogee_visit_in_apstar_inserted = 0
    with database.atomic():
        for chunk in chunked(only_ingest_visits, batch_size):
            n_apogee_visit_in_apstar_inserted += len(
                ApogeeVisitSpectrumInApStar
                .insert_many(chunk)
                .on_conflict(
                    conflict_target=[
                        ApogeeVisitSpectrumInApStar.release,
                        ApogeeVisitSpectrumInApStar.apred,
                        ApogeeVisitSpectrumInApStar.apstar,
                        ApogeeVisitSpectrumInApStar.obj,
                        ApogeeVisitSpectrumInApStar.telescope,
                        ApogeeVisitSpectrumInApStar.healpix,
                        ApogeeVisitSpectrumInApStar.field,
                        ApogeeVisitSpectrumInApStar.prefix,
                        ApogeeVisitSpectrumInApStar.plate,
                        ApogeeVisitSpectrumInApStar.mjd,
                        ApogeeVisitSpectrumInApStar.fiber,
                    ],
                    preserve=(
                        ApogeeVisitSpectrumInApStar.drp_spectrum_pk,
                    )
                )
                .on_conflict(
                    conflict_target=[ApogeeVisitSpectrumInApStar.drp_spectrum_pk],
                    #action="update"
                    action="ignore"
                )                    
                .returning(ApogeeVisitSpectrumInApStar.pk)
                .execute()
            )
            queue.put(dict(advance=len(chunk)))

    queue.put(Ellipsis)
        
    return (n_apogee_visit_in_apstar_inserted, failed_to_match_to_drp_spectrum_pk)


def _get_apstar_metadata(
    apstar,
    keys=(
        "SIMPLE", 
        "FIELD",
        "MEANFIB", 
        "SNR", 
        "SIGFIB", 
        "VSCATTER", 
        "STARFLAG", 
        "NVISITS", 
        "VHELIO",
        "VRAD",
        "VHBARY",
        "VERR", 
        "VERR_MED", 
        "SFILE?",
        "FIBER?"
    ), 
):

    K = len(keys)
    keys_str = "|".join([f"({k})" for k in keys])

    # 80 chars per line, 150 lines -> 12000
    # (12 lines/visit * 100 visits + 100 lines typical header) * 80 -> 104,000
    command_template = " | ".join([
        'hexdump -n 100000 -e \'80/1 "%_p" "\\n"\' {path} 2>/dev/null', # 2>/dev/null suppresses error messages but keeps what we need
        f'egrep "{keys_str}"',
    ])
    commands = f"{command_template.format(path=apstar.absolute_path)}\n"
    
    try:
        outputs = subprocess.check_output(
            commands, 
            shell=True, 
            text=True,
            stderr=subprocess.STDOUT            
        )
    except:
        return { apstar.spectrum_pk: None }
    
    outputs = outputs.strip().split("\n")

    metadata = {}
    for line in outputs:
        try:
            key, value = line.split("=")
            key, value = (key.strip(), value.split()[0].strip(" '"))
        except (IndexError, ValueError): # binary data, probably
            continue
        
        if key in metadata:
            log.warning(f"Multiple key `{key}` found in {apstar}: {expand_path(apstar.path)}")
        metadata[key] = value
    
    return { apstar.spectrum_pk: metadata }

def _migrate_dithered_metadata(pk, absolute_path):
    command = " | ".join([
        f'hexdump -n 12000 -e \'80/1 "%_p" "\\n"\' {absolute_path} 2>/dev/null', # 2>/dev/null suppresses error messages but keeps what we need
        f'egrep "NAXIS1"',
    ])
    try:
        outputs = subprocess.check_output(command, shell=True, text=True)
    except:
        return (pk, None)
        
    if outputs:
        _, naxis1 = outputs.strip(" /\n").split("=")
        # @Nidever: "if there’s 2048 then it hasn’t been dithered, if it’s 4096 then it’s dithered."
        dithered = (int(naxis1) == 4096)
    else:
        # file not found, or corrupted
        dithered = None
    return (pk, dithered)

def migrate_dithered_metadata(
    max_workers=64, 
    batch_size=1000,
    queue=None,
    limit=None
):
    from astra.models.apogee import ApogeeVisitSpectrum
    queue = queue or NoQueue()

    q = (
        ApogeeVisitSpectrum
        .select()
        .where(
            ApogeeVisitSpectrum.dithered.is_null()
        |   ApogeeVisitSpectrum.flag_missing_or_corrupted_file
        )
        .limit(limit)
    )

    update = []
    queue.put(dict(description="Scraping APOGEE visit spectra headers", total=q.count(), completed=0))
    with concurrent.futures.ProcessPoolExecutor(max_workers) as executor:
        futures, spectra, total = ({}, {}, 0)
        for s in q.iterator():
            futures[s.pk] = executor.submit(_migrate_dithered_metadata, s.pk, s.absolute_path)
            spectra[s.pk] = s
            queue.put(dict(advance=1))
            total += 1

        queue.put(dict(description="Parsing APOGEE visit spectra headers", total=total, completed=0))
        for future in concurrent.futures.as_completed(futures.values()):
            pk, dithered = future.result()
            s = spectra.pop(pk)
            s.dithered = dithered
            s.flag_missing_or_corrupted_file = (dithered is None)
            update.append(s)
            queue.put(dict(advance=1))

    n = 0
    if update:
        n += (
            ApogeeVisitSpectrum
            .bulk_update(
                update, 
                fields=[
                    ApogeeVisitSpectrum.dithered,
                    ApogeeVisitSpectrum.spectrum_flags
                ],
                batch_size=batch_size
            )
        )
    queue.put(Ellipsis)
    return n


def migrate_apogee_coadds(apred: str, queue=None, batch_size: int = 1000, limit=None, incremental=True):

    from astra.models.apogee import ApogeeVisitSpectrum, ApogeeCoaddedSpectrumInApStar
    from astra.models.base import database
    from astra.models.source import Source
    from astra.migrations.sdss5db.apogee_drpdb import Star, Visit, RvVisit
    from astra.migrations.sdss5db.catalogdb import (Catalog, CatalogdbModel)

    class SDSS_ID_Flat(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_flat"
            
    class SDSS_ID_Stacked(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_stacked"

    queue = queue or NoQueue()

    max_star_pk = 0
    if incremental:
        max_star_pk = (
            ApogeeCoaddedSpectrumInApStar
            .select(fn.MAX(ApogeeCoaddedSpectrumInApStar.star_pk))
            .scalar() or 0
        )

    # In continuous operations mode, the APOGEE DRP does not update the `star` table to have unique `star_pk`,
    # so we have to sub-query to get the most recent co-add.
    sq = (
        Star
        .select(
            Star.apogee_id,
            Star.telescope,
            fn.MAX(Star.starver).alias("max")
        )
        .where(
            (Star.apred_vers == apred)
        &   (Star.pk > max_star_pk)
        )
        .group_by(Star.apogee_id, Star.telescope)
    )

    q = (
        Star
        .select(
            Star.pk.alias("star_pk"),
            Star.apred_vers.alias("apred"),

            SQL("'sdss5'").alias("release"),
            SQL("'apStar'").alias("filetype"),
            SQL("'stars'").alias("apstar"),

            Star.obj,
            Star.telescope,
            Star.healpix,
            fn.Substr(Star.file, 1, 2).alias("prefix"),

            Star.mjdbeg.alias("min_mjd"),
            Star.mjdend.alias("max_mjd"),
            Star.starver,
            Star.nvisits.alias("n_visits"),
            Star.ngoodvisits.alias("n_good_visits"),
            Star.ngoodrvs.alias("n_good_rvs"),
            Star.snr,
            Star.starflag.alias("spectrum_flags"),
            Star.meanfib.alias("mean_fiber"),
            Star.sigfib.alias("std_fiber"),
            Star.vrad.alias("v_rad"),
            Star.verr.alias("e_v_rad"),
            Star.vscatter.alias("std_v_rad"),
            Star.vmederr.alias("median_e_v_rad"),
            Star.rv_teff.alias("doppler_teff"),
            Star.rv_tefferr.alias("doppler_e_teff"),
            Star.rv_logg.alias("doppler_logg"),
            Star.rv_loggerr.alias("doppler_e_logg"),
            Star.rv_feh.alias("doppler_fe_h"),
            Star.rv_feherr.alias("doppler_e_fe_h"),
            Star.chisq.alias("doppler_rchi2"),
            Star.n_components,
            Star.rv_ccpfwhm.alias("ccfwhm"),
            Star.rv_autofwhm.alias("autofwhm"),
            Star.catalogid,
            Star.gaia_sourceid,
            Star.gaia_release,
            Star.sdss_id,
            Star.jmag.alias("j_mag"),
            Star.jerr.alias("e_j_mag"),
            Star.hmag.alias("h_mag"),
            Star.herr.alias("e_h_mag"),
            Star.kmag.alias("k_mag"),
            Star.kerr.alias("e_k_mag"),
            Star.sdss5_target_catalogids,
            SDSS_ID_Stacked.ra_sdss_id.alias("ra"),
            SDSS_ID_Stacked.dec_sdss_id.alias("dec"),
            SDSS_ID_Stacked.catalogid21,
            SDSS_ID_Stacked.catalogid25,
            SDSS_ID_Stacked.catalogid31,
        )
        .distinct(Star.obj, Star.telescope)
        .join(
            sq, 
            on=(
                (Star.apogee_id == sq.c.apogee_id)
            &   (Star.telescope == sq.c.telescope)
            &   (Star.starver == sq.c.max)
            )
        )
        .switch(Star)
        .join(SDSS_ID_Stacked, JOIN.LEFT_OUTER, on=(Star.sdss_id == SDSS_ID_Stacked.sdss_id))
        .where(
            (Star.apred_vers == apred)
        &   (Star.pk > max_star_pk)
        )
        .limit(limit)
        .dicts()
    )

    source_keys = (
        "catalogid", 
        "catalogid21",
        "catalogid25",
        "catalogid31",
        "sdss_id", 
        "gaia_sourceid", 
        "gaia_release", 
        "sdss5_target_catalogids",
        "ra", 
        "dec", 
        "healpix", 
        "j_mag", 
        "e_j_mag", 
        "h_mag", 
        "e_h_mag", 
        "k_mag", 
        "e_k_mag"
    )    
    source_data, spectrum_data, sdss_id_to_catalogids = separate_spectrum_and_source_data(q, source_keys, queue, f"Parsing APOGEE {apred} coadd spectra")
    sdss_id_to_source_pk = { sdss_id: pk for pk, sdss_id in Source.select(Source.pk, Source.sdss_id).tuples().iterator() }
    new_source_data = { 
        r["sdss_id"]: r 
        for r in source_data if (
            r["sdss_id"] is not None 
        and r["sdss_id"] > 0
        and r["sdss_id"] not in sdss_id_to_source_pk
        ) 
    }

    # Let's create or assign source primary keys.
    n_new_sources = len(new_source_data)
    if n_new_sources > 0:
        queue.put(dict(description=f"Upserting APOGEE {apred} coadd sources", total=n_new_sources, completed=0))
        with database.atomic():
            for chunk in chunked(new_source_data.values(), batch_size):
                (
                    Source
                    .insert_many(chunk)
                    .on_conflict_ignore()
                    .execute()
                )
                queue.put(dict(advance=batch_size))

    # Get the new source primary keys.
    q = (
        Source
        .select(Source.pk, Source.sdss_id)
        .where(Source.sdss_id.in_(list(new_source_data.keys())))
        .tuples()
    )
    sdss_id_to_source_pk.update({ sdss_id: pk for pk, sdss_id in q.iterator() })

    # Assign source primary keys to the spectrum data.
    source_only_keys = set(source_keys) - {"healpix"}
    for r in spectrum_data:
        # Pop out the other fields?
        r["source_pk"] = sdss_id_to_source_pk.get(r["sdss_id"], None)
        for k in source_only_keys: 
            r.pop(k, None)

    preserve = list(
        set(ApogeeCoaddedSpectrumInApStar._meta.fields.values())
    -   {
        ApogeeCoaddedSpectrumInApStar.pk, 
        ApogeeCoaddedSpectrumInApStar.created, 
        ApogeeCoaddedSpectrumInApStar.spectrum_pk,
        ApogeeCoaddedSpectrumInApStar.source_pk,
    }
    )
    n_updated_coadd_spectra = 0
    if spectrum_data:
        queue.put(dict(description=f"Upserting APOGEE {apred} coadded spectra", total=len(spectrum_data), completed=0))
        for chunk in chunked(spectrum_data, batch_size):
            q = (
                ApogeeCoaddedSpectrumInApStar
                .insert_many(chunk)
                .returning(ApogeeCoaddedSpectrumInApStar.pk)
                .on_conflict(
                    conflict_target=[
                        ApogeeCoaddedSpectrumInApStar.release, 
                        ApogeeCoaddedSpectrumInApStar.apred,
                        ApogeeCoaddedSpectrumInApStar.apstar,
                        ApogeeCoaddedSpectrumInApStar.obj,
                        ApogeeCoaddedSpectrumInApStar.telescope,
                        ApogeeCoaddedSpectrumInApStar.field,
                        ApogeeCoaddedSpectrumInApStar.prefix,                        
                    ],
                    preserve=preserve,
                    # These `where` conditions are the only scenarios where we would consider the spectrum as `modified`.
                    where=(
                        (EXCLUDED.starver > ApogeeCoaddedSpectrumInApStar.starver)
                    )
                )
                .tuples()
                .execute()
            )
            for pk in q:
                n_updated_coadd_spectra += 1
            queue.put(dict(advance=batch_size))

    n_new_coadd_spectra = assign_spectrum_pks(ApogeeCoaddedSpectrumInApStar, batch_size, queue)

    return (n_new_sources, n_new_coadd_spectra, n_updated_coadd_spectra)


def migrate_apogee_visits(apred: str, max_mjd: Optional[int] = None, queue=None, batch_size: int = 1000, limit=None, incremental=True):

    from astra.models.apogee import ApogeeVisitSpectrum, ApogeeCoaddedSpectrumInApStar
    from astra.models.base import database
    from astra.models.source import Source
    from astra.migrations.sdss5db.apogee_drpdb import Star, Visit, RvVisit
    from astra.migrations.sdss5db.catalogdb import (Catalog, CatalogdbModel)

    class SDSS_ID_Flat(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_flat"
            
    class SDSS_ID_Stacked(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_stacked"
    queue = queue or NoQueue()

    max_rv_visit_pk, max_visit_pk = (0, 0)
    if incremental:
        max_rv_visit_pk += ApogeeVisitSpectrum.select(fn.MAX(ApogeeVisitSpectrum.rv_visit_pk)).scalar() or 0
        max_visit_pk += ApogeeVisitSpectrum.select(fn.MAX(ApogeeVisitSpectrum.spectrum_pk)).scalar() or 0
    
    if max_mjd is None:
        max_mjd = 1_000_000
        

    # Ingest most recent RV measurements for each star.
    # TODO: Should this really be by `starver`, or should we do it by `created`?
    ssq = (
        RvVisit
        .select(
            RvVisit.visit_pk,
            fn.MAX(RvVisit.starver).alias("max")
        )
        .where(
            (RvVisit.apred_vers == apred)
        &   (RvVisit.pk > max_rv_visit_pk)
        &   (RvVisit.mjd <= max_mjd)
        )
        .group_by(RvVisit.visit_pk)
        .order_by(RvVisit.visit_pk.desc())
    )
    sq = (
        RvVisit
        .select(
            RvVisit.pk,
            RvVisit.visit_pk,
            RvVisit.star_pk,
            RvVisit.bc,
            RvVisit.vrel,
            RvVisit.vrelerr,
            RvVisit.vrad,
            RvVisit.chisq,
            RvVisit.rv_teff,
            RvVisit.rv_tefferr,
            RvVisit.rv_logg,
            RvVisit.rv_loggerr,
            RvVisit.rv_feh,
            RvVisit.rv_feherr,
            RvVisit.xcorr_vrel,
            RvVisit.xcorr_vrelerr,
            RvVisit.xcorr_vrad,
            RvVisit.n_components,
        )
        .join(
            ssq, 
            on=(
                (RvVisit.visit_pk == ssq.c.visit_pk)
            &   (RvVisit.starver == ssq.c.max)
            )
        )
    )

    q = (
        Visit.select(
            Visit.apred,
            Visit.mjd,
            Visit.plate,
            Visit.telescope,
            Visit.field,
            Visit.fiber,
            Visit.file,
            Visit.obj,
            Visit.pk.alias("visit_pk"),
            Visit.dateobs.alias("date_obs"),
            Visit.jd,
            Visit.exptime,
            Visit.nframes.alias("n_frames"),
            Visit.assigned,
            Visit.on_target,
            Visit.valid,
            Visit.snr,
            Visit.starflag.alias("spectrum_flags"),
            Visit.ra.alias("input_ra"),
            Visit.dec.alias("input_dec"),

            # Most recent radial velocity measurement.
            sq.c.bc,
            sq.c.vrel.alias("v_rel"),
            sq.c.vrelerr.alias("e_v_rel"),
            sq.c.vrad.alias("v_rad"),
            sq.c.chisq.alias("doppler_rchi2"),
            sq.c.rv_teff.alias("doppler_teff"),
            sq.c.rv_tefferr.alias("doppler_e_teff"),
            sq.c.rv_logg.alias("doppler_logg"),
            sq.c.rv_loggerr.alias("doppler_e_logg"),
            sq.c.rv_feh.alias("doppler_fe_h"),
            sq.c.rv_feherr.alias("doppler_e_fe_h"),
            sq.c.xcorr_vrel.alias("xcorr_v_rel"),
            sq.c.xcorr_vrelerr.alias("xcorr_e_v_rel"),
            sq.c.xcorr_vrad.alias("xcorr_v_rad"),
            sq.c.n_components,
            sq.c.pk.alias("rv_visit_pk"),
            sq.c.star_pk.alias("star_pk"),

            # Source information,
            Visit.catalogid,
            Visit.sdss_id,
            Visit.healpix,
            Visit.sdss5_target_catalogids,
            Visit.ra_sdss_id.alias("ra"),
            Visit.dec_sdss_id.alias("dec"),
            Visit.gaia_sourceid,
            Visit.gaia_release,
            SDSS_ID_Stacked.catalogid21,
            SDSS_ID_Stacked.catalogid25,
            SDSS_ID_Stacked.catalogid31,

            Visit.jmag.alias("j_mag"),
            Visit.jerr.alias("e_j_mag"),            
            Visit.hmag.alias("h_mag"),
            Visit.herr.alias("e_h_mag"),
            Visit.kmag.alias("k_mag"),
            Visit.kerr.alias("e_k_mag"),
        )
        .distinct(Visit.apred, Visit.mjd, Visit.plate, Visit.telescope, Visit.field, Visit.fiber)
        .join(sq, JOIN.LEFT_OUTER, on=(Visit.pk == sq.c.visit_pk))
        .switch(Visit)
        .join(SDSS_ID_Stacked, JOIN.LEFT_OUTER, on=(Visit.sdss_id == SDSS_ID_Stacked.sdss_id))
        .where(
            (Visit.apred == apred)
        &   (Visit.pk > max_visit_pk)
        &   (Visit.mjd <= max_mjd)
        )
        .limit(limit)
        .dicts()    
    )

    # For each visit, pop out the source information and assign a source ID.
    source_keys = (
        "catalogid", 
        "catalogid21",
        "catalogid25",
        "catalogid31",
        "sdss_id", 
        "gaia_sourceid", 
        "gaia_release", 
        "sdss5_target_catalogids", 
        "ra", 
        "dec", 
        "healpix", 
        "j_mag", 
        "e_j_mag", 
        "h_mag", 
        "e_h_mag", 
        "k_mag", 
        "e_k_mag"
    )
    source_data, spectrum_data, sdss_id_to_catalogids = separate_spectrum_and_source_data(q, source_keys, queue, f"Parsing APOGEE {apred} visit spectra")

    sdss_id_to_source_pk = { sdss_id: pk for pk, sdss_id in Source.select(Source.pk, Source.sdss_id).tuples().iterator() }
    new_source_data = { 
        r["sdss_id"]: r 
        for r in source_data if (
            r["sdss_id"] is not None 
        and r["sdss_id"] > 0
        and r["sdss_id"] not in sdss_id_to_source_pk
        ) 
    }

    # Let's create or assign source primary keys.
    n_new_sources = len(new_source_data)
    if n_new_sources > 0:
        queue.put(dict(description=f"Upserting APOGEE {apred} visit sources", total=n_new_sources, completed=0))
        with database.atomic():
            for chunk in chunked(new_source_data.values(), batch_size):
                (
                    Source
                    .insert_many(chunk)
                    .on_conflict_ignore()
                    .execute()
                )
                queue.put(dict(advance=min(batch_size, len(chunk))))

    # Get the new source primary keys.
    q = (
        Source
        .select(Source.pk, Source.sdss_id)
        .where(Source.sdss_id.in_(list(new_source_data.keys())))
        .tuples()
    )
    sdss_id_to_source_pk.update({ sdss_id: pk for pk, sdss_id in q.iterator() })

    # Assign source primary keys to the spectrum data.
    for r in spectrum_data:
        # Pop out the other fields?
        r.update(
            dict(
                release="sdss5", 
                source_pk=sdss_id_to_source_pk.get(r["sdss_id"], None),
                prefix=r.pop("file").lstrip()[:2], 
                plate=r["plate"].lstrip()
            )
        )
        for k in source_keys[1:]: # all except catalogid
            r.pop(k, None)

    preserve = list(
        set(ApogeeVisitSpectrum._meta.fields.values())
    -   {
            ApogeeVisitSpectrum.pk, 
            ApogeeVisitSpectrum.created, 
            ApogeeVisitSpectrum.spectrum_pk,
            ApogeeVisitSpectrum.source_pk,
        }
    )
    n_updated_visit_spectra = 0
    if spectrum_data:            
        queue.put(dict(description=f"Upserting APOGEE {apred} visit spectra", total=len(spectrum_data), completed=0))
        for chunk in chunked(spectrum_data, batch_size):
            q = (
                ApogeeVisitSpectrum
                .insert_many(chunk)
                .returning(ApogeeVisitSpectrum.pk)
                .on_conflict(
                    conflict_target=[
                        ApogeeVisitSpectrum.release, 
                        ApogeeVisitSpectrum.apred,
                        ApogeeVisitSpectrum.mjd,
                        ApogeeVisitSpectrum.plate,
                        ApogeeVisitSpectrum.telescope,
                        ApogeeVisitSpectrum.field,
                        ApogeeVisitSpectrum.fiber,
                        ApogeeVisitSpectrum.prefix,
                        ApogeeVisitSpectrum.reduction
                    ],
                    preserve=preserve,
                    # These `where` conditions are the only scenarios where we would consider the spectrum as `modified`.
                    where=(
                        (ApogeeVisitSpectrum.rv_visit_pk.is_null() & EXCLUDED.rv_visit_pk.is_null(False))   # New RV measurement; none before.
                    |   (ApogeeVisitSpectrum.rv_visit_pk.is_null(False) & EXCLUDED.rv_visit_pk.is_null())   # Old RV measurement was bad.
                    |   (EXCLUDED.rv_visit_pk > ApogeeVisitSpectrum.rv_visit_pk)                            # Updated RV measurement.
                    )
                )
                .tuples()
                .execute()
            )
            for pk in q:
                n_updated_visit_spectra += 1
            queue.put(dict(advance=min(batch_size, len(chunk))))

    n_new_visit_spectra = assign_spectrum_pks(ApogeeVisitSpectrum, batch_size, queue)

    return (len(new_source_data), n_new_visit_spectra, n_updated_visit_spectra)



def separate_spectrum_and_source_data(q, source_keys, queue, description, k=1000):
    source_data, spectrum_data, sdss_id_to_catalogids = ([], [], {})
    total = q.count()
    if total > 0:
        queue.put(dict(description=description, total=total, completed=0))
        for i, r in enumerate(q.iterator()):
            source = {k: r.get(k) for k in source_keys}
            dr = source.pop("gaia_release")
            gaia_sourceid = source.pop("gaia_sourceid")
            assert dr in ("dr3", "dr2", "", None)
            if dr not in ("", None) and gaia_sourceid is not None and gaia_sourceid > 0:
                source[f"gaia_{dr}_source_id"] = gaia_sourceid

            catalogids = source.pop("sdss5_target_catalogids")

            if source["sdss_id"] is not None and source["sdss_id"] > 0:
                sdss_id = source["sdss_id"]
                sdss_id_to_catalogids.setdefault(sdss_id, set())
                if catalogids:
                    sdss_id_to_catalogids[sdss_id].update(set(map(int, catalogids.split(","))))
                for k in (21, 25, 31):
                    sdss_id_to_catalogids[sdss_id].add(source[f"catalogid{k}"])
            
            source_data.append(source)
            spectrum_data.append(r)   
            if i > 0 and i % k == 0:
                queue.put(dict(advance=k))

    return (source_data, spectrum_data, sdss_id_to_catalogids)


def assign_spectrum_pks(model, batch_size, queue):
    q = (
        model
        .select(model.pk)
        .where(model.spectrum_pk.is_null())
        .tuples()
    )
    n = 0
    if q:
        queue.put(dict(description="Assigning spectrum primary keys", total=q.count(), completed=0))
        for batch in chunked(q, batch_size):
            n += (
                model
                .update(
                    spectrum_pk=Case(None, [
                        (model.pk == pk, spectrum_pk) for spectrum_pk, pk in enumerate_new_spectrum_pks(batch)
                    ])
                )
                .where(model.pk.in_(batch))
                .execute()
            )
            queue.put(dict(advance=min(batch_size, len(batch))))
    return n


def migrate_sdss4_dr17_apogee_spectra_from_sdss5_catalogdb(batch_size: Optional[int] = 100, limit: Optional[int] = None, queue=None):
    """
    Migrate all SDSS4 DR17 APOGEE spectra (`apVisit` and `apStar` files) stored in the SDSS-V database.
    
    :param batch_size: [optional]
        The batch size to use when upserting data.
    
    :returns:
        A tuple of new spectrum identifiers (`astra.models.apogee.ApogeeVisitSpectrum.spectrum_id`)
        that were inserted.
    """
    from astra.models.apogee import ApogeeVisitSpectrum, Spectrum, ApogeeVisitSpectrumInApStar, ApogeeCoaddedSpectrumInApStar
    from astra.models.source import Source
    from astra.models.base import database    
    
    if queue is None:
        queue = NoQueue()

    from astra.migrations.sdss5db.catalogdb import (
        Catalog,
        CatalogToGaia_DR3,
        CatalogToGaia_DR2,
        CatalogdbModel,
        SDSS_DR17_APOGEE_Allvisits as Visit,
    )
    
    class Star(CatalogdbModel):
        class Meta:
            table_name = "allstar_dr17_synspec_rev1"
    
    class CatalogToStar(CatalogdbModel):
        class Meta:
            table_name = "catalog_to_allstar_dr17_synspec_rev1"

    class SDSS_ID_Flat(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_flat"
            
    class SDSS_ID_Stacked(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_stacked"
        
    # Get source-level information first.
    q = (
        Star
        .select(
            Catalog.catalogid,
            Catalog.ra,
            Catalog.dec,
            Catalog.lead,
            Catalog.version_id.alias("version_id"),
            SDSS_ID_Stacked.sdss_id,
            SDSS_ID_Stacked.catalogid21,
            SDSS_ID_Stacked.catalogid25,
            SDSS_ID_Stacked.catalogid31,
            SDSS_ID_Flat.n_associated,
            CatalogToGaia_DR2.target.alias("gaia_dr2_source_id"),
            CatalogToGaia_DR3.target.alias("gaia_dr3_source_id"),
            Star.memberflag.alias("sdss4_apogee_member_flags"),
            Star.apogee_target1.alias("sdss4_apogee_target1_flags"),
            Star.apogee_target2.alias("sdss4_apogee_target2_flags"),
            Star.apogee2_target1.alias("sdss4_apogee2_target1_flags"),
            Star.apogee2_target2.alias("sdss4_apogee2_target2_flags"),
            Star.apogee2_target3.alias("sdss4_apogee2_target3_flags"),
            Star.apogee_id.alias("sdss4_apogee_id"),            
            Star.extratarg.alias("sdss4_apogee_extra_target_flags")
        )
        .join(CatalogToStar, JOIN.LEFT_OUTER, on=(CatalogToStar.target_id == Star.apstar_id))        
        .join(Catalog, JOIN.LEFT_OUTER, on=(CatalogToStar.catalogid == Catalog.catalogid))        
        .join(
            SDSS_ID_Flat, 
            JOIN.LEFT_OUTER, 
            on=(
                (SDSS_ID_Flat.catalogid == Catalog.catalogid)
            &   (SDSS_ID_Flat.rank == 1)
            )
        )
        .join(SDSS_ID_Stacked, JOIN.LEFT_OUTER, on=(SDSS_ID_Stacked.sdss_id == SDSS_ID_Flat.sdss_id))
        .join(CatalogToGaia_DR2, JOIN.LEFT_OUTER, on=(CatalogToGaia_DR2.catalog == Catalog.catalogid))
        .join(CatalogToGaia_DR3, JOIN.LEFT_OUTER, on=(CatalogToGaia_DR3.catalog == CatalogToGaia_DR2.catalogid))
        .limit(limit)
        .dicts()
    )
    
    total = limit or q.count()
    queue.put(dict(total=total, completed=0, description="Querying APOGEE dr17 sources"))
    source_data = {}
    for row in q.iterator():
        source_key = row["sdss4_apogee_id"]            
        if source_key in source_data:
            # Take the minimum sdss_id
            if row["sdss_id"] is not None and source_data[source_key]["sdss_id"] is not None:
                source_data[source_key]["sdss_id"] = min(source_data[source_key]["sdss_id"], row["sdss_id"])
            else:
                source_data[source_key]["sdss_id"] = source_data[source_key]["sdss_id"] or row["sdss_id"]
            for key, value in row.items():
                # Merge any targeting keys
                if key.startswith("sdss4_apogee") and key.endswith("_flags"):
                    source_data[source_key][key] |= value                    
        else:
            source_data[source_key] = row

        queue.put(dict(advance=1))

    # Assign the Sun to have SDSS_ID = 0, because it's very special to me.
    source_data["VESTA"]["sdss_id"] = 0
                            
    # Upsert the sources
    with database.atomic():
        queue.put(dict(description="Upserting APOGEE dr17 sources", completed=0, total=len(source_data)))

        for chunk in chunked(source_data.values(), batch_size):
            (
                Source
                .insert_many(chunk)
                .on_conflict(
                    # if we conflict over sdss_id or gaia_dr3_source_id, then
                    # at least update the sdss4_apogee_id
                    conflict_target=[Source.gaia_dr3_source_id],
                    preserve=(Source.sdss4_apogee_id, )
                )
                .on_conflict(
                    conflict_target=[Source.gaia_dr2_source_id],
                    preserve=(Source.sdss4_apogee_id, )
                )
                .on_conflict(
                    conflict_target=[Source.sdss_id],
                    preserve=(Source.sdss4_apogee_id, )
                )
                .on_conflict_ignore()
                .execute()
            )
            n = min(batch_size, len(chunk))
            queue.put(dict(advance=n))

    q = (
        Source
        .select(
            Source.pk,
            Source.sdss_id,
        )
        .tuples()
        .iterator()
    )

    # Need to be able to look up source_pks given a target_id
    lookup_source_pk_given_sdss_id = { sdss_id: pk for pk, sdss_id in q }
    lookup_source_pk_given_sdss4_apogee_id = {}
    queue.put(dict(description="Linking APOGEE dr17 sources", total=len(source_data), completed=0))
    for sdss4_apogee_id, attrs in tqdm(source_data.items()):
        try:

            # This can happ
            source_pk = lookup_source_pk_given_sdss_id[attrs["sdss_id"]]
        except KeyError:
            try:
                # this can happen when the sdss4_apogee_id changed mid survey
                # eg AP00430387+4118048
                source_pk = Source.get(sdss4_apogee_id=attrs["sdss4_apogee_id"]).pk
            except:
                source_pk = Source.get(gaia_dr3_source_id=attrs["gaia_dr3_source_id"]).pk
        lookup_source_pk_given_sdss4_apogee_id[sdss4_apogee_id] = source_pk
        queue.put(dict(advance=1))
    
    q = (
        Visit
        .select(
            Visit.mjd,
            Visit.plate,
            Visit.telescope,
            Visit.field,
            Visit.apogee_id.alias("obj"), # see notes in astra.models.apogee.ApogeeVisitSpectrum about this
            Visit.fiberid.alias("fiber"),
            Visit.jd,
            Visit.dateobs.alias("date_obs"),
            Visit.starflag.alias("spectrum_flags"),
            Visit.ra.alias("input_ra"),
            Visit.dec.alias("input_dec"),
            Visit.snr,
            Visit.file,

            # Radial velocity information
            Visit.vrel.alias("v_rel"),
            Visit.vrelerr.alias("e_v_rel"),
            Visit.vhelio.alias("v_rad"),
            Visit.bc,
            Visit.rv_teff.alias("doppler_teff"),
            Visit.rv_logg.alias("doppler_logg"),
            Visit.rv_feh.alias("doppler_fe_h"),
            Visit.xcorr_vrel.alias("xcorr_v_rel"),
            Visit.xcorr_vrelerr.alias("xcorr_e_v_rel"),
            Visit.xcorr_vhelio.alias("xcorr_v_rad"),
            Visit.rv_chi2.alias("doppler_rchi2"),
            Visit.ccfwhm,
            Visit.autofwhm,
            Visit.n_components,
            Visit.rv_flag.alias("doppler_flags"),
        )
        .limit(limit)
        .dicts()
    )
    
    apogee_visit_spectra = []
    queue.put(dict(total=q.count(), completed=0, description="Querying APOGEE dr17 visit spectra"))
    for row in q.iterator():
        basename = row.pop("file")
        row["plate"] = row["plate"].lstrip()        
        if row["telescope"] == "apo1m":
            row["reduction"] = row["obj"]
        
        queue.put(dict(advance=1))

        try:
            source_pk = lookup_source_pk_given_sdss4_apogee_id[row['obj']]        
        except KeyError:
            if limit is None:
                raise
            else:
                continue
        else:
            apogee_visit_spectra.append({
                "source_pk": source_pk,
                "release": "dr17",
                "apred": "dr17",
                "prefix": basename.lstrip()[:2],
                **row
            })

    # Upsert the spectra
    pks = upsert_many(
        ApogeeVisitSpectrum,
        ApogeeVisitSpectrum.pk,
        apogee_visit_spectra,
        batch_size,
        queue,
        "Upserting APOGEE dr17 visit spectra"
    )

    # Assign spectrum_pk values to any spectra missing it.
    N = len(pks)
    if pks:
        queue.put(dict(total=N, completed=0, description="Assigning primary keys to spectra"))
        N_assigned = 0
        for batch in chunked(pks, batch_size):
            stuff = []
            for spectrum_pk, pk in  enumerate_new_spectrum_pks(batch):
                stuff.append((ApogeeVisitSpectrum.pk == pk, spectrum_pk))
                #for spectrum_pk, pk in

            B =  (
                ApogeeVisitSpectrum
                .update(
                    spectrum_pk=Case(None, stuff)                
                )
                .where(ApogeeVisitSpectrum.pk.in_(batch))
                .execute()
            )
            queue.put(dict(advance=B))
            N_assigned += B

        #log.info(f"There were {N} spectra inserted and we assigned {N_assigned} spectra with new spectrum_pk values")

    # Sanity check
    q = flatten(
        ApogeeVisitSpectrum
        .select(ApogeeVisitSpectrum.pk)
        .where(ApogeeVisitSpectrum.spectrum_pk.is_null())
        .tuples()
    )
    if q:
        queue.put(dict(description="Sanity checking APOGEE dr17 spectrum primary keys", total=len(q), completed=0))
        N_updated = 0
        for batch in chunked(q, batch_size):
            stuff = []
            for spectrum_pk, pk in enumerate_new_spectrum_pks(batch):
                stuff.append((ApogeeVisitSpectrum.pk == pk, spectrum_pk))

            n = (
                ApogeeVisitSpectrum
                .update(
                    spectrum_pk=Case(None, stuff)
                )
                .where(ApogeeVisitSpectrum.pk.in_(batch))
                .execute()            
            )
            queue.put(dict(advance=n))
            N_updated += n
        #log.warning(f"Assigned spectrum_pks to {N_updated} existing spectra")

    assert not (
        ApogeeVisitSpectrum
        .select(ApogeeVisitSpectrum.pk)
        .where(ApogeeVisitSpectrum.spectrum_pk.is_null())
        .exists()
    )    

    # Ingest ApogeeCoadded    
    
    q = (
        Star
        .select(
            Star.apogee_id.alias("obj"),
            Star.field,
            Star.telescope,
            Star.nvisits.alias("n_visits"),
        )
        .distinct(Star.apogee_id, Star.field, Star.telescope)
        .limit(limit)
    )

    # Need to get source_pks based on existing 
    apogee_coadded_spectra = []
    #for star in tqdm(q.iterator(), total=limit or q.count()):
    queue.put(dict(description="Querying APOGEE dr17 coadded spectra", total=q.count(), completed=0))
    for star in q.iterator():
        try:
            source_pk = lookup_source_pk_given_sdss4_apogee_id[star.obj]             
        except KeyError:
            if limit is None:
                raise
            else:
                queue.put(dict(advance=1))
                continue
        else:
            apogee_coadded_spectra.append(
                dict(
                    source_pk=source_pk,
                    release="dr17",
                    filetype="apStar",
                    apred="dr17",
                    apstar="stars",
                    obj=star.obj,
                    telescope=star.telescope,
                    field=star.field,
                    prefix="ap" if star.telescope.startswith("apo") else "as",
                )
            )
            queue.put(dict(advance=1))

    # Upsert the spectra
    pks = upsert_many(
        ApogeeCoaddedSpectrumInApStar,
        ApogeeCoaddedSpectrumInApStar.pk,
        apogee_coadded_spectra,
        batch_size,
        queue,
        "Upserting APOGEE dr17 coadded spectra"
    )

    # Assign spectrum_pk values to any spectra missing it.
    N = len(pks)
    if pks:
        #with tqdm(total=N, desc="Assigning primary keys to spectra") as pb:
        queue.put(dict(description="Assigning primary keys to spectra", total=N, completed=0))
        N_assigned = 0
        for batch in chunked(pks, batch_size):
            cases = []
            for spectrum_pk, pk in enumerate_new_spectrum_pks(batch):
                cases.append((ApogeeCoaddedSpectrumInApStar.pk == pk, spectrum_pk))

            B = (
                ApogeeCoaddedSpectrumInApStar
                .update(spectrum_pk=Case(None, cases))
                .where(ApogeeCoaddedSpectrumInApStar.pk.in_(batch))
                .execute()
            )
            queue.put(dict(advance=B))
            N_assigned += B
        #log.info(f"There were {N} spectra inserted and we assigned {N_assigned} spectra with new spectrum_pk values")
    
    queue.put(Ellipsis)
    
    return None
    