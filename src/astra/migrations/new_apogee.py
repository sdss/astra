
from peewee import JOIN, chunked, Case, fn, SQL, EXCLUDED
from astra.migrations.utils import enumerate_new_spectrum_pks, upsert_many, NoQueue
from tqdm import tqdm
from subprocess import check_output
import concurrent.futures
from astra.utils import log



def migrate_apogee_spectra_from_sdss5_apogee_drpdb(apred: str, queue=None, limit=None, incremental=True, **kwargs):
    queue = queue or NoQueue()

    # Let's do visits first.
    v_n_new_sources, v_n_new_spectra, v_n_updated_spectra = migrate_apogee_visits(apred, queue=queue, limit=limit, incremental=incremental, **kwargs)
    
    # Now do co-added spectra.
    c_n_new_sources, c_n_new_spectra, c_n_updated_spectra = migrate_apogee_coadds(apred, queue=queue, limit=limit, incremental=incremental, **kwargs)
    
    # Now we need to check all the apStar files for the ApogeeVisitSpectrumInApStar entries,
    # and for whether it was dithered or not.
    queue.put(Ellipsis)
    log.info(f"APOGEE {apred}: {v_n_new_spectra} new visits; {v_n_updated_spectra} updated visits; {v_n_new_sources} new sources")
    log.info(f"APOGEE {apred}: {c_n_new_spectra} new coadds; {c_n_updated_spectra} updated coadds; {c_n_new_sources} new sources")
    
    return None

def _migrate_dithered_metadata(pk, absolute_path):
    command = " | ".join([
        f'hexdump -n 12000 -e \'80/1 "%_p" "\\n"\' {absolute_path} 2>/dev/null', # 2>/dev/null suppresses error messages but keeps what we need
        f'egrep "NAXIS1"',
    ])
    outputs = check_output(command, shell=True, text=True)
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


def migrate_apogee_visits(apred: str, queue=None, batch_size: int = 1000, limit=None, incremental=True):

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