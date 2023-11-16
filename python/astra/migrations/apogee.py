import concurrent
import subprocess
import numpy as np

from collections import OrderedDict
from peewee import chunked, Case, fn, JOIN, IntegrityError
from typing import Optional
from tqdm import tqdm
from astropy.table import Table
from astra.models.apogee import ApogeeVisitSpectrum, Spectrum, ApogeeVisitSpectrumInApStar, ApogeeCoaddedSpectrumInApStar
from astra.models.source import Source
from astra.models.base import database
from astra.utils import expand_path, flatten, log

from astra.migrations.utils import enumerate_new_spectrum_pks, upsert_many


def copy_doppler_results_from_visit_to_coadd(batch_size: Optional[int] = 100, limit: Optional[int] = None):

    q = (
        ApogeeCoaddedSpectrumInApStar
        .select()
        .where(ApogeeCoaddedSpectrumInApStar.doppler_teff.is_null())
    )

    N_updated = 0
    total = limit or q.count()
    with tqdm(total=total, desc="Updating", unit="star") as pb:
        for chunk in chunked(q.iterator(), batch_size):
            sources = {}
            for spectrum in chunk:
                sources.setdefault(spectrum.source_pk, [])
                sources[spectrum.source_pk].append(spectrum)

            q_visit = (
                ApogeeVisitSpectrum
                .select()
                .where(ApogeeVisitSpectrum.source_pk.in_([s.source_pk for s in chunk]))
            )

            updated = []
            for visit in q_visit:
                for spectrum in sources[visit.source_pk]:
                    spectrum.doppler_teff   = float(visit.doppler_teff or np.nan)
                    spectrum.doppler_e_teff = float(visit.doppler_e_teff or np.nan)
                    spectrum.doppler_logg   = float(visit.doppler_logg  or np.nan)
                    spectrum.doppler_e_logg = float(visit.doppler_e_logg or np.nan)
                    spectrum.doppler_fe_h   = float(visit.doppler_fe_h  or np.nan)
                    spectrum.doppler_e_fe_h = float(visit.doppler_e_fe_h or np.nan)
                    spectrum.doppler_rchi2  = float(visit.doppler_rchi2  or np.nan)
                    spectrum.doppler_flags  = visit.doppler_flags 
                    updated.append(spectrum)
            
                        
            N_updated += (
                ApogeeCoaddedSpectrumInApStar
                .bulk_update(
                    updated,
                    fields=[
                        ApogeeCoaddedSpectrumInApStar.doppler_teff,
                        ApogeeCoaddedSpectrumInApStar.doppler_e_teff,
                        ApogeeCoaddedSpectrumInApStar.doppler_logg,
                        ApogeeCoaddedSpectrumInApStar.doppler_e_logg,
                        ApogeeCoaddedSpectrumInApStar.doppler_fe_h,
                        ApogeeCoaddedSpectrumInApStar.doppler_e_fe_h,
                        ApogeeCoaddedSpectrumInApStar.doppler_rchi2,
                        ApogeeCoaddedSpectrumInApStar.doppler_flags,
                    ]
                )
            )
            
            pb.update(min(len(chunk), batch_size))
    return N_updated



def migrate_apogee_obj_from_source(batch_size: Optional[int] = 100, limit: Optional[int] = None):

    q = (
        ApogeeVisitSpectrum
        .select(
            ApogeeVisitSpectrum.spectrum_pk,
            Source.sdss4_apogee_id
        )
        .join(Source, on=(ApogeeVisitSpectrum.source_id == Source.id))
        .where(
            ApogeeVisitSpectrum.obj.is_null()
        &   ApogeeVisitSpectrum.healpix.is_null() # don't overwrite the apogee_drp-computed healpix, even if it's wrong
        &   Source.sdss4_apogee_id.is_null(False)
        )
        .tuples()
    )

    total = limit or q.count()
    with tqdm(total=total, desc="Updating", unit="spectra") as pb:
        for chunk in chunked(q.iterator(), batch_size):
            objs = { spectrum_pk: obj for spectrum_pk, obj in chunk }
            q = (
                ApogeeVisitSpectrum
                .select()
                .where(ApogeeVisitSpectrum.spectrum_pk.in_(list(objs.keys())))
            )
            batch = list(q)
            for spectrum in batch:
                spectrum.obj = objs[spectrum.spectrum_pk]
            
            pb.update(
                ApogeeVisitSpectrum
                .bulk_update(
                    batch,
                    fields=[ApogeeVisitSpectrum.obj]
                )
            )

    return pb.n


def _migrate_apstar_metadata(
        apstars,
        keys=(
            "NWAVE", 
            "FIELD",
            "MEANFIB", 
            "SNR", 
            "SIGFIB", 
            "VSCATTER", 
            "STARFLAG", 
            "NVISITS", 
            "VHELIO",
            "VERR", 
            "VERR_MED", 
            "SFILE?",
            "FIBER?"
        ), 
    ):

    #keys = ("MEANFIB", "SNR", "SIGFIB", "STARFLAGS", "NVISITS", "VHELIO", "VERR", "VERR_MED", "SFILE?", "DATE?" )
    K = len(keys)
    keys_str = "|".join([f"({k})" for k in keys])

    # 80 chars per line, 150 lines -> 12000
    # (12 lines/visit * 100 visits + 100 lines typical header) * 80 -> 104,000
    command_template = " | ".join([
        'hexdump -n 100000 -e \'80/1 "%_p" "\\n"\' {path}',
        f'egrep "{keys_str}"',
        #f"head -n {K}"
    ])
    commands = ""
    for apstar in apstars:
        path = expand_path(apstar.path)
        commands += f"{command_template.format(path=path)}\n"
    
    try:
        outputs = subprocess.check_output(commands, shell=True, text=True)
    except subprocess.CalledProcessError:        
        return {}
    
    outputs = outputs.strip().split("\n")

    p, all_metadata = (-1, {})
    for line in outputs:
        try:
            key, value = line.split("=")
            key, value = (key.strip(), value.split()[0].strip(" '"))
        except (IndexError, ValueError): # binary data, probably
            continue
        
        if key == "NWAVE":
            p += 1
        spectrum_pk = apstars[p].spectrum_pk
        all_metadata.setdefault(spectrum_pk, {})
        if key in all_metadata[spectrum_pk]:
            log.warning(f"Multiple key `{key}` found in {apstars[p]}: {expand_path(apstars[p].path)}")
            raise a
        all_metadata[spectrum_pk][key] = value
    
    return all_metadata


def _get_apstar_metadata(
        apstar,
        keys=(
            "NWAVE", 
            "FIELD",
            "MEANFIB", 
            "SNR", 
            "SIGFIB", 
            "VSCATTER", 
            "STARFLAG", 
            "NVISITS", 
            "VHELIO",
            "VRAD",
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
        'hexdump -n 100000 -e \'80/1 "%_p" "\\n"\' {path}',
        f'egrep "{keys_str}"',
    ])
    path = expand_path(apstar.path)
    commands = f"{command_template.format(path=path)}\n"
    
    try:
        outputs = subprocess.check_output(
            commands, 
            shell=True, 
            text=True,
            stderr=subprocess.STDOUT            
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
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


        
def fix_version_id_edge_cases(version_ids=(13, 24, )):
    from astra.migrations.sdss5db.catalogdb import Catalog, CatalogdbModel, CatalogToGaia_DR3

    class SDSS_ID_Flat(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_flat"
            
    class SDSS_ID_Stacked(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_stacked"
    
    q = (
        Source
        .select()
        .where(Source.version_id.in_(version_ids))
    )

    failed = []
    for updated, source in enumerate(q, start=1):
        
        # Get new catalog information based on gaia_dr3_source_id
        q_source = (
            Catalog
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
            )
            .join(SDSS_ID_Flat, on=(SDSS_ID_Flat.catalogid == Catalog.catalogid))
            .join(SDSS_ID_Stacked, on=(SDSS_ID_Stacked.sdss_id == SDSS_ID_Flat.sdss_id))
            .switch(Catalog)
            .join(CatalogToGaia_DR3, on=(CatalogToGaia_DR3.catalog == Catalog.catalogid))
            .where(CatalogToGaia_DR3.target == source.gaia_dr3_source_id)
            .order_by(SDSS_ID_Flat.sdss_id.asc())
            .dicts()
            .first()
        )
        if q_source is not None:                            
            for k, v in q_source.items():
                setattr(source, k, v)                
            try:
                source.save()
            except IntegrityError:
                failed.append(source)
                
                log.info(f"Failed to update {source} with sdss_id={source.sdss_id}. Updating dependencies.")
                existing_source_pk = Source.get(sdss_id=source.sdss_id).pk
                for expr, field in source.dependencies():
                    for item in field.model.select().where(expr):
                        log.info(f"\t{field.model} {item} source_pk={existing_source_pk}")
                        item.source_pk = existing_source_pk
                        item.save()
                
                log.info(f"Deleting {source}")
                source.delete_instance()
            
        else:
            log.warning(f"Could not find updated source for {source}")
        
    return (updated, failed)


def migrate_sdss4_dr17_apogee_spectra_from_sdss5_catalogdb(batch_size: Optional[int] = 100, limit: Optional[int] = None, max_workers: Optional[int] = 8):
    """
    Migrate all SDSS4 DR17 APOGEE spectra (`apVisit` and `apStar` files) stored in the SDSS-V database.
    
    :param batch_size: [optional]
        The batch size to use when upserting data.
    
    :returns:
        A tuple of new spectrum identifiers (`astra.models.apogee.ApogeeVisitSpectrum.spectrum_id`)
        that were inserted.
    """
    
    from astra.migrations.sdss5db.catalogdb import (
        Catalog,
        SDSS_DR17_APOGEE_Allvisits as Visit,
        CatalogToGaia_DR3,
        CatalogToGaia_DR2,
        CatalogdbModel
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
        

    log.info(f"Migrating SDSS4 DR17 apStar spectra from SDSS5 catalog database")

    # Get source-level information first.
    log.info(f"Ingesting source-level information")
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
        )
        .join(CatalogToStar, JOIN.LEFT_OUTER, on=(CatalogToStar.target_id == Star.apstar_id))        
        .join(Catalog, JOIN.LEFT_OUTER, on=(CatalogToStar.catalogid == Catalog.catalogid))        
        .join(SDSS_ID_Flat, JOIN.LEFT_OUTER, on=(SDSS_ID_Flat.catalogid == Catalog.catalogid))
        .join(SDSS_ID_Stacked, JOIN.LEFT_OUTER, on=(SDSS_ID_Stacked.sdss_id == SDSS_ID_Flat.sdss_id))
        .join(CatalogToGaia_DR2, JOIN.LEFT_OUTER, on=(CatalogToGaia_DR2.catalog == Catalog.catalogid))
        .join(CatalogToGaia_DR3, JOIN.LEFT_OUTER, on=(CatalogToGaia_DR3.catalog == CatalogToGaia_DR2.catalogid))
        .dicts()
    )

    source_data = {}
    for row in tqdm(q.iterator(), total=1):
        source_key = row["sdss4_apogee_id"]            
        if source_key in source_data:
            # Take the minimum sdss_id
            source_data[source_key]["sdss_id"] = min(source_data[source_key]["sdss_id"], row["sdss_id"])
            for key, value in row.items():
                # Merge any targeting keys
                if key.startswith("sdss4_apogee") and key.endswith("_flags"):
                    source_data[source_key][key] |= value                    
        else:
            source_data[source_key] = row

    # Assign the Sun to have SDSS_ID = 0, because it's very special to me.
    source_data["VESTA"]["sdss_id"] = 0
                            
    # Upsert the sources
    with database.atomic():
        with tqdm(desc="Upserting sources", total=len(source_data)) as pb:
            for chunk in chunked(source_data.values(), batch_size):
                (
                    Source
                    .insert_many(chunk)
                    .on_conflict_ignore()
                    .execute()
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()

    log.info(f"Getting data for sources")
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
    for sdss4_apogee_id, attrs in source_data.items():
        source_pk = lookup_source_pk_given_sdss_id[attrs["sdss_id"]]
        lookup_source_pk_given_sdss4_apogee_id[sdss4_apogee_id] = source_pk
    
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
    for row in tqdm(q.iterator(), total=limit or 1, desc="Retrieving spectra"):
        basename = row.pop("file")
        row["plate"] = row["plate"].lstrip()        
        if row["telescope"] == "apo1m":
            row["reduction"] = row["obj"]
        
        source_pk = lookup_source_pk_given_sdss4_apogee_id[row['obj']]        
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
        desc="Upserting spectra"
    )

    # Assign spectrum_pk values to any spectra missing it.
    N = len(pks)
    if pks:
        with tqdm(total=N, desc="Assigning primary keys to spectra") as pb:
            N_assigned = 0
            for batch in chunked(pks, batch_size):
                B =  (
                    ApogeeVisitSpectrum
                    .update(
                        spectrum_pk=Case(None, (
                            (ApogeeVisitSpectrum.pk == pk, spectrum_pk) for spectrum_pk, pk in enumerate_new_spectrum_pks(batch)
                        ))
                    )
                    .where(ApogeeVisitSpectrum.pk.in_(batch))
                    .execute()
                )
                pb.update(B)
                N_assigned += B

        log.info(f"There were {N} spectra inserted and we assigned {N_assigned} spectra with new spectrum_pk values")

    # Sanity check
    q = flatten(
        ApogeeVisitSpectrum
        .select(ApogeeVisitSpectrum.pk)
        .where(ApogeeVisitSpectrum.spectrum_pk.is_null())
        .tuples()
    )
    if q:
        N_updated = 0
        for batch in chunked(q, batch_size):
            N_updated += (
                ApogeeVisitSpectrum
                .update(
                    spectrum_pk=Case(None, [
                        (ApogeeVisitSpectrum.pk == pk, spectrum_pk) for spectrum_pk, pk in enumerate_new_spectrum_pks(batch)
                    ])
                )
                .where(ApogeeVisitSpectrum.pk.in_(batch))
                .execute()            
            )
        log.warning(f"Assigned spectrum_pks to {N_updated} existing spectra")

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
    for star in tqdm(q.iterator(), total=limit or q.count()):
        source_pk = lookup_source_pk_given_sdss4_apogee_id[star.obj]             
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

    # Upsert the spectra
    pks = upsert_many(
        ApogeeCoaddedSpectrumInApStar,
        ApogeeCoaddedSpectrumInApStar.pk,
        apogee_coadded_spectra,
        batch_size,
        desc="Upserting spectra"
    )

    # Assign spectrum_pk values to any spectra missing it.
    N = len(pks)
    if pks:
        with tqdm(total=N, desc="Assigning primary keys to spectra") as pb:
            N_assigned = 0
            for batch in chunked(pks, batch_size):
                B = (
                    ApogeeCoaddedSpectrumInApStar
                    .update(
                        spectrum_pk=Case(None, (
                            (ApogeeCoaddedSpectrumInApStar.pk == pk, spectrum_pk) for spectrum_pk, pk in enumerate_new_spectrum_pks(batch)
                        ))
                    )
                    .where(ApogeeCoaddedSpectrumInApStar.pk.in_(batch))
                    .execute()
                )
                pb.update(B)
                N_assigned += B

        log.info(f"There were {N} spectra inserted and we assigned {N_assigned} spectra with new spectrum_pk values")
    else:
        log.info(f"No new spectra inserted")
    
    migrate_sdss4_dr17_metadata_from_headers()
    
    return None
    
def migrate_sdss4_dr17_metadata_from_headers():

    executor = concurrent.futures.ProcessPoolExecutor(max_workers)
    q = (
        ApogeeCoaddedSpectrumInApStar
        .select()
        .where((ApogeeCoaddedSpectrumInApStar.release == "dr17"))
        .iterator()
    )

    spectra, futures = ({}, [])
    with tqdm(total=limit or 0, desc="Retrieving metadata", unit="spectra") as pb:
        for total, spectrum in enumerate(q, start=1):
            futures.append(executor.submit(_get_apstar_metadata, spectrum))
            spectra[spectrum.spectrum_pk] = spectrum
            pb.update()

    failed_spectrum_pks, apogee_visit_spectra_in_apstar = ([], [])
    with tqdm(total=total, desc="Collecting results", unit="spectra") as pb:
        for future in concurrent.futures.as_completed(futures):
            for spectrum_pk, metadata in future.result().items():
                if metadata is None:
                    failed_spectrum_pks.append(spectrum_pk)
                    continue
                    
                spectrum = spectra[spectrum_pk]

                mjds = []
                sfiles = [metadata[f"SFILE{i}"] for i in range(1, int(metadata["NVISITS"]) + 1)]
                for sfile in sfiles:
                    if spectrum.telescope == "apo1m":
                        #"$SAS_BASE_DIR/dr17/apogee/spectro/redux/{apred}/visit/{telescope}/{field}/{mjd}/apVisit-{apred}-{mjd}-{reduction}.fits"
                        # sometimes it is stored as a float AHGGGGHGGGGHGHGHGH
                        mjds.append(int(float(sfile.split("-")[2])))
                    else:
                        mjds.append(int(float(sfile.split("-")[3])))
                        # "$SAS_BASE_DIR/dr17/apogee/spectro/redux/{apred}/visit/{telescope}/{field}/{plate}/{mjd}/{prefix}Visit-{apred}-{plate}-{mjd}-{fiber:0>3}.fits"

                assert len(sfiles) == int(metadata["NVISITS"])
                
                spectrum.snr = float(metadata["SNR"])
                spectrum.mean_fiber = float(metadata["MEANFIB"])
                spectrum.std_fiber = float(metadata["SIGFIB"])
                spectrum.n_good_visits = int(metadata["NVISITS"])
                spectrum.n_good_rvs = int(metadata["NVISITS"])
                spectrum.v_rad = float(metadata["VHELIO"])
                spectrum.e_v_rad = float(metadata["VERR"])
                spectrum.std_v_rad = float(metadata["VSCATTER"])
                spectrum.median_e_v_rad = float(metadata["VERR_MED"])
                spectrum.spectrum_flags = metadata["STARFLAG"]
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
                    field=spectrum.field,
                    prefix=spectrum.prefix,
                    reduction=spectrum.obj if spectrum.telescope == "apo1m" else None           
                )
                for i, (mjd, sfile) in enumerate(zip(mjds, sfiles), start=1):
                    if spectrum.telescope != "apo1m":
                        plate = sfile.split("-")[2]
                    else:
                        # plate not known..
                        plate = metadata["FIELD"].strip()

                    kwds = star_kwds.copy()
                    kwds.update(
                        mjd=mjd,
                        fiber=int(metadata[f"FIBER{i}"]),
                        plate=plate
                    )
                    apogee_visit_spectra_in_apstar.append(kwds)
                
                pb.update()
                
    if len(failed_spectrum_pks) > 0:
        log.warning(f"There were {len(failed_spectrum_pks)} spectra that we could not parse headers from")

    with tqdm(total=total, desc="Updating", unit="spectra") as pb:     
        for chunk in chunked(spectra.values(), batch_size):
            pb.update(
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
            )

    log.info(f"Creating visit spectra")
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
        .where(ApogeeVisitSpectrum.release == "dr17")
        .tuples()
    )
    drp_spectrum_data = {}
    for obj, spectrum_pk, telescope, plate, mjd, fiber in tqdm(q.iterator(), desc="Getting DRP spectrum data"):
        drp_spectrum_data.setdefault(obj, {})
        key = "_".join(map(str, (telescope, plate, mjd, fiber)))
        drp_spectrum_data[obj][key] = spectrum_pk

    log.info(f"Matching to DRP spectra")

    for spectrum_pk, visit in enumerate_new_spectrum_pks(apogee_visit_spectra_in_apstar):
        key = "_".join(map(str, [visit[k] for k in ("telescope", "plate", "mjd", "fiber")]))
        visit.update(
            spectrum_pk=spectrum_pk,
            drp_spectrum_pk=drp_spectrum_data[visit["obj"]][key]
        )

    with database.atomic():
        with tqdm(desc="Upserting visit spectra", total=len(apogee_visit_spectra_in_apstar)) as pb:
            for chunk in chunked(apogee_visit_spectra_in_apstar, batch_size):
                (
                    ApogeeVisitSpectrumInApStar
                    .insert_many(chunk)
                    .on_conflict_ignore()
                    .execute()
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()
    return None


def fix_apvisit_instances_of_invalid_gaia_dr3_source_id(fuzz_ratio_min=75):

    source_pks_up_for_deletion = []

    from fuzzywuzzy import fuzz
    from astra.migrations.sdss5db.catalogdb import (
        Catalog,
        SDSS_DR17_APOGEE_Allvisits as Visit,
        SDSS_DR17_APOGEE_Allstarmerge as Star,
        CatalogToGaia_DR3,
        CatalogToGaia_DR2,
        CatalogToTwoMassPSC,
        TwoMassPSC,
        CatalogdbModel
    )

    class SDSS_ID_Flat(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_flat"
            
    class SDSS_ID_Stacked(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_stacked"

    # Fix any instances where gaia_dr3_source_id = 0
    q = (
        ApogeeVisitSpectrum
        .select()
        .join(Source, on=(ApogeeVisitSpectrum.source_pk == Source.pk))
        .where(
            (Source.gaia_dr3_source_id <= 0) | (Source.gaia_dr3_source_id.is_null())
        )
    )
    N_broken = q.count()
    log.warning(f"Trying to fix {N_broken} instances where gaia_dr3_source_id <= 0 or NULL. This could take a few minutes.")

    N_fixed = 0
    for record in tqdm(q.iterator(), total=N_broken):

        sdss4_apogee_id = record.source.sdss4_apogee_id or record.obj
        if sdss4_apogee_id.startswith("2M") or sdss4_apogee_id.startswith("AP"):
            designation = sdss4_apogee_id[2:]
        else:
            designation = sdss4_apogee_id

        q = (
            CatalogToTwoMassPSC
            .select(CatalogToTwoMassPSC.catalog)
            .join(TwoMassPSC, on=(TwoMassPSC.pts_key == CatalogToTwoMassPSC.target))
            .where(TwoMassPSC.designation == designation)
            .tuples()
            .first()
        )
        if q:
            catalogid, = q
            q_identifiers = (
                Catalog
                .select(
                    Catalog.ra,
                    Catalog.dec,
                    Catalog.catalogid,
                    Catalog.version_id.alias("version_id"),
                    Catalog.lead,
                    SDSS_ID_Flat.sdss_id,
                    SDSS_ID_Flat.n_associated,
                    SDSS_ID_Stacked.catalogid21,
                    SDSS_ID_Stacked.catalogid25,
                    SDSS_ID_Stacked.catalogid31,     
                    CatalogToGaia_DR2.target.alias("gaia_dr2_source_id"),
                    CatalogToGaia_DR3.target.alias("gaia_dr3_source_id"),
                )
                .join(SDSS_ID_Flat, JOIN.LEFT_OUTER, on=(Catalog.catalogid == SDSS_ID_Flat.catalogid))
                .join(SDSS_ID_Stacked, JOIN.LEFT_OUTER, on=(SDSS_ID_Stacked.sdss_id == SDSS_ID_Flat.sdss_id))
                .switch(Catalog)
                .join(CatalogToGaia_DR3, JOIN.LEFT_OUTER, on=(SDSS_ID_Stacked.catalogid31 == CatalogToGaia_DR3.catalog))
                .switch(Catalog)
                .join(CatalogToGaia_DR2, JOIN.LEFT_OUTER, on=(SDSS_ID_Stacked.catalogid31 == CatalogToGaia_DR2.catalog))
                .where(Catalog.catalogid == catalogid)
                .dicts()
                .first()
            )

            # Update this source
            for key, value in q_identifiers.items():
                setattr(record.source, key, value)

            try:
                record.source.sdss4_apogee_id = sdss4_apogee_id
                record.source.save()

            except IntegrityError as exception:
                log.exception(f"Unable to update record {record} with source {record.source}: {exception}")

                # In these situations, there are usually two different APOGEE_ID values which are nominally the same:
                # e.g., 2M17204208+6538238 and J17204208+6538238
                # and then we try to assign the same SDSS_ID value to two different APOGEE_ID values.

                # If this is the case, let's assign things to the other source because it will have more information.
                alt_source = Source.get(sdss_id=record.source.sdss_id)
                alt_sdss4_apogee_id = alt_source.sdss4_apogee_id

                fuzz_ratio = fuzz.ratio(sdss4_apogee_id, alt_sdss4_apogee_id)
                if fuzz_ratio > fuzz_ratio_min:
                    
                    # Delete the alternative source>
                    source_pks_up_for_deletion.append(record.source.pk)

                    record.source_pk = alt_source.pk
                    record.save()

                    N_fixed += 1

                else:
                    raise RuntimeError(f"record {record} with source={record.source} not matched {sdss4_apogee_id} != {alt_sdss4_apogee_id} ({fuzz_ratio} > {fuzz_ratio_min})")
            else:
                N_fixed += 1

    log.warning(f"Tried to fix {N_fixed} of {N_broken} examples")
    if source_pks_up_for_deletion:
        log.warning(f"Source primary keys up for deletion: {source_pks_up_for_deletion}")
        N_deleted = (
            Source
            .delete()
            .where(
                Source.pk.in_(tuple(set(source_pks_up_for_deletion)))
            )
            .execute()
        )
        log.warning(f"Deleted {N_deleted} sources")

    return (N_fixed, N_broken)


def migrate_new_apstar_only_from_sdss5_apogee_drpdb(
    apred: str, limit: Optional[int] = None,
    batch_size: Optional[int] = 1000
):
    
    from astra.migrations.sdss5db.apogee_drpdb import Star, Visit, RvVisit
    from astra.migrations.sdss5db.catalogdb import CatalogdbModel

    # Get SDSS identifiers
    q = (
        Source
        .select(
            Source.pk,
            Source.catalogid,
            Source.catalogid21,
            Source.catalogid25,
            Source.catalogid31
        )        
    )
    source_pks_by_catalogid = {}
    for pk, *catalogids in q.tuples().iterator():
        for catalogid in catalogids:
            if catalogid is not None and catalogid > 0:
                source_pks_by_catalogid[catalogid] = pk
            
    q = (
        Star
        .select(
            Star.obj,
            Star.telescope,
            Star.healpix,
            Star.mjdbeg.alias("min_mjd"),
            Star.mjdend.alias("max_mjd"),
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
            # TODO: doppler_flags? 
            Star.n_components,
            Star.rv_ccpfwhm.alias("ccfwhm"),
            Star.rv_autofwhm.alias("autofwhm"),
            Star.pk.alias("star_pk"),
            Star.catalogid,
            #SDSS_ID_Flat.sdss_id,
        )
        .distinct(Star.pk)
        #.join(SDSS_ID_Flat, on=(Star.catalogid == SDSS_ID_Flat.catalogid))
        .where(Star.apred_vers == apred)
    )
    if limit is not None:
        q = q.limit(limit)

    # q = q.order_by(SDSS_ID_Flat.sdss_id.asc())    
    spectrum_data = {}
    unknown_stars = []
    for star in tqdm(q.dicts().iterator(), total=1, desc="Getting spectra"):

        star.update(
            release="sdss5",
            filetype="apStar",
            apstar="stars",
            apred=apred,
        )
        # This part assumes that we have already ingested everything from the visits, but that might not be true (e.g., when testing things).
        # TODO: create sources if we dont have them
        #sdss_id = star.pop("sdss_id")
        
        # Sometimes we can get here and there is a source catalogid that we have NEVER seen before, even if we have ingested
        # ALL the visits. The rason is because there are "no good visits". That's fucking annoying, because even if they aren't
        # "good visits", they should appear in the visit table.
        catalogid = star["catalogid"]
        try:
            star["source_pk"] = source_pks_by_catalogid[catalogid]
        except KeyError:
            unknown_stars.append(star)
        else:        
            star.pop("catalogid")
            spectrum_data[star["star_pk"]] = star

    if len(unknown_stars) > 0:
        log.warning(f"There were {len(unknown_stars)} unknown stars")
        for star in unknown_stars:
            if star["n_good_visits"] != 0:
                log.warning(f"Star {star['obj']} (catalogid={star['catalogid']}; star_pk={star['star_pk']} has {star['n_good_visits']} good visits but we've never seen it before")

    star_pks = []
    with database.atomic():
        with tqdm(desc="Upserting", total=len(spectrum_data)) as pb:
            for chunk in chunked(spectrum_data.values(), batch_size):
                star_pks.extend(
                    flatten(
                        ApogeeCoaddedSpectrumInApStar
                        .insert_many(chunk)                        
                        .on_conflict(
                            conflict_target=[
                                ApogeeCoaddedSpectrumInApStar.star_pk,
                            ],
                            preserve=(
                                ApogeeCoaddedSpectrumInApStar.min_mjd,
                                ApogeeCoaddedSpectrumInApStar.max_mjd,
                                ApogeeCoaddedSpectrumInApStar.n_visits,
                                ApogeeCoaddedSpectrumInApStar.n_good_visits,
                                ApogeeCoaddedSpectrumInApStar.n_good_rvs,
                                ApogeeCoaddedSpectrumInApStar.snr,
                                ApogeeCoaddedSpectrumInApStar.spectrum_flags,
                                ApogeeCoaddedSpectrumInApStar.mean_fiber,
                                ApogeeCoaddedSpectrumInApStar.std_fiber,
                                ApogeeCoaddedSpectrumInApStar.v_rad,
                                ApogeeCoaddedSpectrumInApStar.e_v_rad,
                                ApogeeCoaddedSpectrumInApStar.std_v_rad,
                                ApogeeCoaddedSpectrumInApStar.median_e_v_rad,
                                ApogeeCoaddedSpectrumInApStar.doppler_teff,
                                ApogeeCoaddedSpectrumInApStar.doppler_e_teff,
                                ApogeeCoaddedSpectrumInApStar.doppler_logg,
                                ApogeeCoaddedSpectrumInApStar.doppler_e_logg,
                                ApogeeCoaddedSpectrumInApStar.doppler_fe_h,
                                ApogeeCoaddedSpectrumInApStar.doppler_e_fe_h,
                                ApogeeCoaddedSpectrumInApStar.doppler_rchi2,
                                ApogeeCoaddedSpectrumInApStar.n_components,
                                ApogeeCoaddedSpectrumInApStar.ccfwhm,
                                ApogeeCoaddedSpectrumInApStar.autofwhm,
                            )
                        )
                        .returning(ApogeeCoaddedSpectrumInApStar.pk)
                        .tuples()
                        .execute()
                    )
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()

    return star_pks


def migrate_new_apvisit_from_sdss5_apogee_drpdb(
    apred: str,
    since,
    limit: Optional[int] = None,
    batch_size: Optional[int] = 1000,
    max_workers: Optional[int] = 32
):
    
    from astra.migrations.sdss5db.apogee_drpdb import Star, Visit, RvVisit
    from astra.migrations.sdss5db.catalogdb import (
        Catalog,
        CatalogToGaia_DR3,
        CatalogToGaia_DR2,
        CatalogdbModel
    )

    class SDSS_ID_Flat(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_flat"
            
    class SDSS_ID_Stacked(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_stacked"

    log.info(f"Migrating SDSS5 apVisit spectra from SDSS5 catalog database")


    ssq = (
        RvVisit
        .select(
            RvVisit.visit_pk,
            fn.MAX(RvVisit.starver).alias("max")
        )
        .where(
            (RvVisit.apred_vers == apred)
        #&   (RvVisit.catalogid > 0) # Some RM_COSMOS fields with catalogid=0 (e.g., apogee_drp.visit = 7494220)
        &   (RvVisit.created >= since)
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
        .where(RvVisit.created >= since)
    )
    if limit is not None:
        sq = sq.limit(limit)    


    q = (
        Visit.select(
            Visit.apred,
            Visit.mjd,
            Visit.plate,
            Visit.telescope,
            Visit.field,
            Visit.fiber,
            Visit.file,
            #Visit.prefix,
            Visit.obj,
            Visit.pk.alias("visit_pk"),
            Visit.dateobs.alias("date_obs"),
            Visit.jd,
            Visit.exptime,
            Visit.nframes.alias("n_frames"),
            Visit.assigned,
            Visit.on_target,
            Visit.valid,
            Visit.starflag.alias("spectrum_flags"),
            Visit.catalogid,
            Visit.ra.alias("input_ra"),
            Visit.dec.alias("input_dec"),

            # Source information,
            Visit.gaiadr2_sourceid.alias("gaia_dr2_source_id"),
            CatalogToGaia_DR3.target_id.alias("gaia_dr3_source_id"),
            #Catalog.catalogid.alias("catalogid"),
            Catalog.version_id.alias("version_id"),
            Catalog.lead,
            Catalog.ra,
            Catalog.dec,
            SDSS_ID_Flat.sdss_id,
            SDSS_ID_Flat.n_associated,
            SDSS_ID_Stacked.catalogid21,
            SDSS_ID_Stacked.catalogid25,
            SDSS_ID_Stacked.catalogid31,
            Visit.jmag.alias("j_mag"),
            Visit.jerr.alias("e_j_mag"),            
            Visit.hmag.alias("h_mag"),
            Visit.herr.alias("e_h_mag"),
            Visit.kmag.alias("k_mag"),
            Visit.kerr.alias("e_k_mag"),
            
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
            sq.c.star_pk.alias("star_pk")
        )
        .distinct(
            Visit.apred,
            Visit.mjd,
            Visit.plate,
            Visit.telescope,
            Visit.field,
            Visit.fiber,
        )
        .join(sq, on=(Visit.pk == sq.c.visit_pk)) #
        .switch(Visit)
        # Need to join by Catalog on the visit catalogid (not gaia DR2) because sometimes Gaia DR2 value is 0
        # Doing it like this means we might end up with some `catalogid` actually NOT being v1, but
        # we will have to fix that afterwards. It will be indicated by the `version_id`.
        .join(Catalog, JOIN.LEFT_OUTER, on=(Catalog.catalogid == Visit.catalogid))
        .switch(Visit)
        .join(CatalogToGaia_DR2, JOIN.LEFT_OUTER, on=(Visit.gaiadr2_sourceid == CatalogToGaia_DR2.target_id))
        .join(CatalogToGaia_DR3, JOIN.LEFT_OUTER, on=(CatalogToGaia_DR2.catalogid == CatalogToGaia_DR3.catalogid))
        .switch(Catalog)
        .join(SDSS_ID_Flat, JOIN.LEFT_OUTER, on=(Catalog.catalogid == SDSS_ID_Flat.catalogid))
        .join(SDSS_ID_Stacked, JOIN.LEFT_OUTER, on=(SDSS_ID_Stacked.sdss_id == SDSS_ID_Flat.sdss_id))
        .where(Visit.created >= since)
        .dicts()
    )
    
    # 
    # TODO: now do after 'since'
    print("DO BAD spectra after 'since'")
    # rv visits with created after 'since'

    # The query above will return the same ApogeeVisit when it is associated with multiple sdss_id values,
    # but the .on_conflict_ignore() when upserting will mean that the spectra are not duplicated in the database.
    source_only_keys = (
        "sdss_id",
        "catalogid21",
        "catalogid25",
        "catalogid31",
        "n_associated",
        "catalogid",
        "gaia_dr2_source_id",
        "gaia_dr3_source_id",
        "version_id",
        "lead",
        "ra",
        "dec",
        "j_mag",
        "e_j_mag",
        "h_mag",
        "e_h_mag",
        "k_mag",
        "e_k_mag",
    )
    

    source_data, spectrum_data, matched_sdss_ids = (OrderedDict(), [], {})
    for row in tqdm(q.iterator(), total=limit or 1, desc="Retrieving spectra"):
        basename = row.pop("file")

        catalogid = row["catalogid"]        
        
        this_source_data = dict(zip(source_only_keys, [row.pop(k) for k in source_only_keys]))

        if catalogid in source_data:
            # make sure the only difference is SDSS_ID
            if this_source_data["sdss_id"] is not None and source_data[catalogid]["sdss_id"] is not None:
                matched_sdss_ids[catalogid] = min(this_source_data["sdss_id"], source_data[catalogid]["sdss_id"])
            elif this_source_data["sdss_id"] is None or source_data[catalogid]["sdss_id"] is None:
                matched_sdss_ids[catalogid] = this_source_data["sdss_id"] or  source_data[catalogid]["sdss_id"]                
            else:
                matched_sdss_ids[catalogid] = None
        else:
            source_data[catalogid] = this_source_data
            matched_sdss_ids[catalogid] = this_source_data["sdss_id"]
        
        row["plate"] = row["plate"].lstrip()
        
        spectrum_data.append({
            "catalogid": catalogid,
            "release": "sdss5",
            "apred": apred,
            "prefix": basename.lstrip()[:2],
            **row
        }) 
        
    q_without_rvs = (
        Visit.select(
            Visit.apred,
            Visit.mjd,
            Visit.plate,
            Visit.telescope,
            Visit.field,
            Visit.fiber,
            Visit.file,
            #Visit.prefix,
            Visit.obj,
            Visit.pk.alias("visit_pk"),
            Visit.dateobs.alias("date_obs"),
            Visit.jd,
            Visit.exptime,
            Visit.nframes.alias("n_frames"),
            Visit.assigned,
            Visit.on_target,
            Visit.valid,
            Visit.starflag.alias("spectrum_flags"),
            Visit.catalogid,
            Visit.ra.alias("input_ra"),
            Visit.dec.alias("input_dec"),

            # Source information,
            Visit.gaiadr2_sourceid.alias("gaia_dr2_source_id"),
            CatalogToGaia_DR3.target_id.alias("gaia_dr3_source_id"),
            #Catalog.catalogid.alias("catalogid"),
            Catalog.version_id.alias("version_id"),
            Catalog.lead,
            Catalog.ra,
            Catalog.dec,
            SDSS_ID_Flat.sdss_id,
            SDSS_ID_Flat.n_associated,
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
        .join(RvVisit, JOIN.LEFT_OUTER, on=(Visit.pk == RvVisit.visit_pk))
        .switch(Visit)
        # Need to join by Catalog on the visit catalogid (not gaia DR2) because sometimes Gaia DR2 value is 0
        # Doing it like this means we might end up with some `catalogid` actually NOT being v1, but
        # we will have to fix that afterwards. It will be indicated by the `version_id`.
        .join(Catalog, JOIN.LEFT_OUTER, on=(Catalog.catalogid == Visit.catalogid))
        .switch(Visit)
        .join(CatalogToGaia_DR2, JOIN.LEFT_OUTER, on=(Visit.gaiadr2_sourceid == CatalogToGaia_DR2.target_id))
        .join(CatalogToGaia_DR3, JOIN.LEFT_OUTER, on=(CatalogToGaia_DR2.catalogid == CatalogToGaia_DR3.catalogid))
        .switch(Catalog)
        .join(SDSS_ID_Flat, JOIN.LEFT_OUTER, on=(Catalog.catalogid == SDSS_ID_Flat.catalogid))
        .join(SDSS_ID_Stacked, JOIN.LEFT_OUTER, on=(SDSS_ID_Stacked.sdss_id == SDSS_ID_Flat.sdss_id))
        .where(RvVisit.pk.is_null() & (Visit.apred_vers == apred) & (Visit.catalogid > 0))
        #.where(Visit.pk == 9025065)
        .dicts()
    )


    for row in tqdm(q_without_rvs.iterator(), total=limit or 1, desc="Retrieving bad spectra"):
        basename = row.pop("file")

        assert row["catalogid"] is not None
        catalogid = row["catalogid"]        
        
        this_source_data = dict(zip(source_only_keys, [row.pop(k) for k in source_only_keys]))

        if catalogid in source_data:
            # make sure the only difference is SDSS_ID
            if this_source_data["sdss_id"] is None or source_data[catalogid]["sdss_id"] is None:
                matched_sdss_ids[catalogid] = (this_source_data["sdss_id"] or source_data[catalogid]["sdss_id"])
            else:    
                matched_sdss_ids[catalogid] = min(this_source_data["sdss_id"], source_data[catalogid]["sdss_id"])
        else:
            source_data[catalogid] = this_source_data
            matched_sdss_ids[catalogid] = this_source_data["sdss_id"]
        
        row["plate"] = row["plate"].lstrip()
        
        spectrum_data.append({
            "catalogid": catalogid, # Will be removed later, just for matching sources.
            "release": "sdss5",
            "apred": apred,
            "prefix": basename.lstrip()[:2],
            **row
        })

        
    
    # Upsert any new sources
    with database.atomic():
        with tqdm(desc="Upserting sources", total=len(source_data)) as pb:
            for chunk in chunked(source_data.values(), batch_size):
                (
                    Source
                    .insert_many(chunk)
                    .on_conflict_ignore()
                    .execute()
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()


    log.info(f"Getting data for sources")
    q = (
        Source
        .select(
            Source.pk,
            Source.sdss_id,
            Source.catalogid,
            Source.catalogid21,
            Source.catalogid25,
            Source.catalogid31
        )
        .tuples()
        .iterator()
    )

    source_pk_by_sdss_id = {}
    source_pk_by_catalogid = {}
    for pk, sdss_id, *catalogids in q:
        if sdss_id is not None:
            source_pk_by_sdss_id[sdss_id] = pk
        for catalogid in catalogids:
            if catalogid is not None:
                source_pk_by_catalogid[catalogid] = pk
        
    for each in spectrum_data:
        catalogid = each["catalogid"]
        try:
            matched_sdss_id = matched_sdss_ids[catalogid]
            source_pk = source_pk_by_sdss_id[matched_sdss_id]            
        except:
            source_pk = source_pk_by_catalogid[catalogid]
            log.warning(f"No SDSS_ID found for catalogid={catalogid}, assigned to source_pk={source_pk}: {each}")

        each["source_pk"] = source_pk
    
    
    # make sure we don't have duplicates
    spectrum_data_as_dict = { ea["visit_pk"]: ea for ea in spectrum_data }
    spectrum_data = list(spectrum_data_as_dict.values())
    
                    
    pks = []
    with database.atomic():
        with tqdm(desc="Upserting", total=len(spectrum_data)) as pb:
            for chunk in chunked(spectrum_data, batch_size):
                pks.extend(
                    flatten(
                        ApogeeVisitSpectrum
                        .insert_many(chunk)                        
                        .on_conflict(
                            conflict_target=[
                                ApogeeVisitSpectrum.visit_pk,
                            ],
                            preserve=(
                                # update `obj` to fix the name discrepancy which caused 200 nights to go missing
                                # otherwise we can't match new spectra to existing DRP spectra because the existing
                                # DRP spectra have their names shortened
                                ApogeeVisitSpectrum.obj, 
                                ApogeeVisitSpectrum.visit_pk, 
                                ApogeeVisitSpectrum.date_obs, 
                                ApogeeVisitSpectrum.jd, 
                                ApogeeVisitSpectrum.exptime, 
                                ApogeeVisitSpectrum.n_frames, 
                                ApogeeVisitSpectrum.assigned, 
                                ApogeeVisitSpectrum.on_target, 
                                ApogeeVisitSpectrum.valid, 
                                ApogeeVisitSpectrum.spectrum_flags, 
                                ApogeeVisitSpectrum.input_ra, 
                                ApogeeVisitSpectrum.input_dec, 
                                ApogeeVisitSpectrum.bc, 
                                ApogeeVisitSpectrum.v_rel, 
                                ApogeeVisitSpectrum.e_v_rel, 
                                ApogeeVisitSpectrum.v_rad, 
                                ApogeeVisitSpectrum.doppler_rchi2, 
                                ApogeeVisitSpectrum.doppler_teff, 
                                ApogeeVisitSpectrum.doppler_e_teff, 
                                ApogeeVisitSpectrum.doppler_logg, 
                                ApogeeVisitSpectrum.doppler_e_logg, 
                                ApogeeVisitSpectrum.doppler_fe_h, 
                                ApogeeVisitSpectrum.doppler_e_fe_h, 
                                ApogeeVisitSpectrum.xcorr_v_rel, 
                                ApogeeVisitSpectrum.xcorr_e_v_rel, 
                                ApogeeVisitSpectrum.xcorr_v_rad, 
                                ApogeeVisitSpectrum.n_components, 
                                ApogeeVisitSpectrum.rv_visit_pk, 
                                ApogeeVisitSpectrum.star_pk, 
                            )
                        )
                        .returning(ApogeeVisitSpectrum.pk)
                        .tuples()
                        .execute()
                    )
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()


    # Assign spectrum_pk values to any spectra missing it.
    N = len(pks)
    if pks:
        with tqdm(total=N, desc="Assigning primary keys to spectra") as pb:
            N_assigned = 0
            for batch in chunked(pks, batch_size):
                
                is_missing = flatten(
                    ApogeeVisitSpectrum
                    .select(
                        ApogeeVisitSpectrum.pk,
                    )
                    .where(
                        ApogeeVisitSpectrum.pk.in_(batch)
                    &   ApogeeVisitSpectrum.spectrum_pk.is_null()
                    )
                    .tuples()
                )                
                
                if is_missing:
                    B =  (
                        ApogeeVisitSpectrum
                        .update(
                            spectrum_pk=Case(None, (
                                (ApogeeVisitSpectrum.pk == pk, spectrum_pk) for spectrum_pk, pk in enumerate_new_spectrum_pks(is_missing)
                            ))
                        )
                        .where(
                            ApogeeVisitSpectrum.pk.in_(is_missing)
                        )
                        .execute()
                    )
                    N_assigned += B
                pb.update(batch_size)
    
        log.info(f"There were {N} spectra: {N - N_assigned} updated; {N_assigned} inserted")
    
    # Sanity check    
    
    # Now migrate the apogee coadded star and apogee visit in star
    
    from astra.migrations.sdss5db.apogee_drpdb import Star, Visit, RvVisit
    from astra.migrations.sdss5db.catalogdb import CatalogdbModel

    # Get SDSS identifiers
    q = (
        Source
        .select(
            Source.pk,
            Source.catalogid,
            Source.catalogid21,
            Source.catalogid25,
            Source.catalogid31
        )        
    )
    source_pks_by_catalogid = {}
    for pk, *catalogids in q.tuples().iterator():
        for catalogid in catalogids:
            if catalogid is not None and catalogid > 0:
                source_pks_by_catalogid[catalogid] = pk
            

    q = (
        Star
        .select(
            Star.obj,
            Star.telescope,
            Star.healpix,
            Star.mjdbeg.alias("min_mjd"),
            Star.mjdend.alias("max_mjd"),
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
            # TODO: doppler_flags? 
            Star.n_components,
            Star.rv_ccpfwhm.alias("ccfwhm"),
            Star.rv_autofwhm.alias("autofwhm"),
            Star.pk.alias("star_pk"),
            Star.catalogid,
            #SDSS_ID_Flat.sdss_id,
        )
        .distinct(Star.pk)
        #.join(SDSS_ID_Flat, on=(Star.catalogid == SDSS_ID_Flat.catalogid))
        .where(Star.apred_vers == apred)
        .where(Star.created >= since)
    )
    if limit is not None:
        q = q.limit(limit)

    # q = q.order_by(SDSS_ID_Flat.sdss_id.asc())
    
    spectrum_data = {}
    unknown_stars = []
    for star in tqdm(q.dicts().iterator(), total=1, desc="Getting spectra"):

        star.update(
            release="sdss5",
            filetype="apStar",
            apstar="stars",
            apred=apred,
        )
        # This part assumes that we have already ingested everything from the visits, but that might not be true (e.g., when testing things).
        # TODO: create sources if we dont have them
        #sdss_id = star.pop("sdss_id")
        
        # Sometimes we can get here and there is a source catalogid that we have NEVER seen before, even if we have ingested
        # ALL the visits. The rason is because there are "no good visits". That's fucking annoying, because even if they aren't
        # "good visits", they should appear in the visit table.
        catalogid = star["catalogid"]
        try:
            star["source_pk"] = source_pks_by_catalogid[catalogid]
        except KeyError:
            unknown_stars.append(star)
        else:        
            star.pop("catalogid")
            spectrum_data[star["star_pk"]] = star

    if len(unknown_stars) > 0:
        log.warning(f"There were {len(unknown_stars)} unknown stars")
        for star in unknown_stars:
            if star["n_good_visits"] != 0:
                log.warning(f"Star {star['obj']} (catalogid={star['catalogid']}; star_pk={star['star_pk']} has {star['n_good_visits']} good visits but we've never seen it before")

                    
    star_pks = []
    with database.atomic():
        with tqdm(desc="Upserting", total=len(spectrum_data)) as pb:
            for chunk in chunked(spectrum_data.values(), batch_size):
                star_pks.extend(
                    flatten(
                        ApogeeCoaddedSpectrumInApStar
                        .insert_many(chunk)                        
                        .on_conflict(
                            conflict_target=[
                                ApogeeCoaddedSpectrumInApStar.star_pk,
                            ],
                            preserve=(
                                ApogeeCoaddedSpectrumInApStar.min_mjd,
                                ApogeeCoaddedSpectrumInApStar.max_mjd,
                                ApogeeCoaddedSpectrumInApStar.n_visits,
                                ApogeeCoaddedSpectrumInApStar.n_good_visits,
                                ApogeeCoaddedSpectrumInApStar.n_good_rvs,
                                ApogeeCoaddedSpectrumInApStar.snr,
                                ApogeeCoaddedSpectrumInApStar.spectrum_flags,
                                ApogeeCoaddedSpectrumInApStar.mean_fiber,
                                ApogeeCoaddedSpectrumInApStar.std_fiber,
                                ApogeeCoaddedSpectrumInApStar.v_rad,
                                ApogeeCoaddedSpectrumInApStar.e_v_rad,
                                ApogeeCoaddedSpectrumInApStar.std_v_rad,
                                ApogeeCoaddedSpectrumInApStar.median_e_v_rad,
                                ApogeeCoaddedSpectrumInApStar.doppler_teff,
                                ApogeeCoaddedSpectrumInApStar.doppler_e_teff,
                                ApogeeCoaddedSpectrumInApStar.doppler_logg,
                                ApogeeCoaddedSpectrumInApStar.doppler_e_logg,
                                ApogeeCoaddedSpectrumInApStar.doppler_fe_h,
                                ApogeeCoaddedSpectrumInApStar.doppler_e_fe_h,
                                ApogeeCoaddedSpectrumInApStar.doppler_rchi2,
                                ApogeeCoaddedSpectrumInApStar.n_components,
                                ApogeeCoaddedSpectrumInApStar.ccfwhm,
                                ApogeeCoaddedSpectrumInApStar.autofwhm,
                            )
                        )
                        .returning(ApogeeCoaddedSpectrumInApStar.pk)
                        .tuples()
                        .execute()
                    )
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()



    # Assign spectrum_pk values to any spectra missing it.
    N = len(star_pks)
    if star_pks:
        with tqdm(total=N, desc="Assigning primary keys to spectra") as pb:
            N_assigned = 0
            for batch in chunked(star_pks, batch_size):
                
                is_missing = flatten(
                    ApogeeCoaddedSpectrumInApStar
                    .select(
                        ApogeeCoaddedSpectrumInApStar.pk,
                    )
                    .where(
                        ApogeeCoaddedSpectrumInApStar.pk.in_(batch)
                    &   ApogeeCoaddedSpectrumInApStar.spectrum_pk.is_null()
                    )
                    .tuples()
                )                            
                if is_missing:
                    B = (
                        ApogeeCoaddedSpectrumInApStar
                        .update(
                            spectrum_pk=Case(None, (
                                (ApogeeCoaddedSpectrumInApStar.pk == pk, spectrum_pk) for spectrum_pk, pk in enumerate_new_spectrum_pks(is_missing)
                            ))
                        )
                        .where(ApogeeCoaddedSpectrumInApStar.pk.in_(is_missing))
                        .execute()
                    )
                    N_assigned += B
                pb.update(batch_size)

        log.info(f"There were {N} spectra: {N - N_assigned} existing; {N_assigned} inserted")
    else:
        log.info(f"No new spectra inserted")

    # Do the ApogeeVisitSpectrumInApStar level things based on the FITS files themselves.
    # This is because the APOGEE DRP makes decisions about 'what gets in' to the apStar files, and this information cannot be inferred from the database (according to Nidever).

    executor = concurrent.futures.ProcessPoolExecutor(max_workers)
    q = (
        ApogeeCoaddedSpectrumInApStar
        .select()
        .where(ApogeeCoaddedSpectrumInApStar.pk.in_(star_pks))
        .iterator()
    )

    spectra, futures, total = ({}, [], 0)
    with tqdm(total=limit or 0, desc="Retrieving metadata", unit="spectra") as pb:
        for spectrum in q:
            futures.append(executor.submit(_get_apstar_metadata, spectrum))
            spectra[spectrum.spectrum_pk] = spectrum
            pb.update()

    visit_spectrum_data = []
    failed_spectrum_pks = []
    with tqdm(total=total, desc="Collecting results", unit="spectra") as pb:
        for future in concurrent.futures.as_completed(futures):
            result = future.result()         
            for spectrum_pk, metadata in result.items():
                    
                if metadata is None:
                    failed_spectrum_pks.append(spectrum_pk)
                    continue
                                
                spectrum = spectra[spectrum_pk]

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
                spectrum.v_rad = float(metadata["VRAD"])
                spectrum.e_v_rad = float(metadata["VERR"])
                spectrum.std_v_rad = float(metadata["VSCATTER"])
                spectrum.median_e_v_rad = float(metadata.get("VERR_MED", np.nan))
                spectrum.spectrum_flags = metadata["STARFLAG"]
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
                
                pb.update()


    with tqdm(total=total, desc="Updating", unit="spectra") as pb:     
        for chunk in chunked(spectra.values(), batch_size):
            pb.update(
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
            )

    log.info(f"Creating visit spectra")
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
        .tuples()
    )
    drp_spectrum_data = {}
    for obj, spectrum_pk, telescope, plate, mjd, fiber in tqdm(q.iterator(), desc="Getting DRP spectrum data"):
        drp_spectrum_data.setdefault(obj, {})
        key = "_".join(map(str, (telescope, plate, mjd, fiber)))
        drp_spectrum_data[obj][key] = spectrum_pk

    log.info(f"Matching to DRP spectra")

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
        
    if len(failed_to_match_to_drp_spectrum_pk) > 0:
        log.warning(f"There were {len(failed_to_match_to_drp_spectrum_pk)} spectra that we could not match to DRP spectra")
        log.warning(f"Example: {failed_to_match_to_drp_spectrum_pk[0]}")

    n_apogee_visit_in_apstar_inserted = 0
    with database.atomic():
        with tqdm(desc="Upserting visit spectra", total=len(only_ingest_visits)) as pb:
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
                        action="ignore"
                    )                    
                    .returning(ApogeeVisitSpectrumInApStar.pk)
                    .execute()
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()    
    
    log.info(f"There were {n_apogee_visit_in_apstar_inserted} ApogeeVisitSpectrumInApStar spectra inserted")
    
    # print: 
    print("NOW DO MIGRATION FOR ANY MISSING PHOTOMETRY, ASTROMETRY, UPDATE N_VISITS, etc")    
    
    return None


def migrate_apvisit_from_sdss5_apogee_drpdb(
    apred: str,
    rvvisit_where: Optional = None,
    batch_size: Optional[int] = 100, 
    limit: Optional[int] = None,
):
    """
    Migrate all new APOGEE visit information (`apVisit` files) stored in the SDSS-V database, which is reported
    by the SDSS-V APOGEE data reduction pipeline.

    :param apred: [optional]
        Limit the ingestion to spectra with a specified `apred` version.
                
    :param batch_size: [optional]
        The batch size to use when upserting data.
    
    :param limit: [optional]
        Limit the ingestion to `limit` spectra.

    :returns:
        A tuple of new spectrum identifiers (`astra.models.apogee.ApogeeVisitSpectrum.spectrum_id`)
        that were inserted.
    """

    from astra.migrations.sdss5db.apogee_drpdb import Star, Visit, RvVisit
    from astra.migrations.sdss5db.catalogdb import (
        Catalog,
        CatalogToGaia_DR3,
        CatalogToGaia_DR2,
        CatalogdbModel
    )

    class SDSS_ID_Flat(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_flat"
            
    class SDSS_ID_Stacked(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_stacked"

    log.info(f"Migrating SDSS5 apVisit spectra from SDSS5 catalog database")


    ssq = (
        RvVisit
        .select(
            RvVisit.visit_pk,
            fn.MAX(RvVisit.starver).alias("max")
        )
        .where(
            (RvVisit.apred_vers == apred)
        &   (RvVisit.catalogid > 0) # Some RM_COSMOS fields with catalogid=0 (e.g., apogee_drp.visit = 7494220)
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
    if rvvisit_where is not None:
        sq = (
            sq
            .where(rvvisit_where)
        )

    if limit is not None:
        sq = sq.limit(limit)

    q = (
        Visit.select(
            Visit.apred,
            Visit.mjd,
            Visit.plate,
            Visit.telescope,
            Visit.field,
            Visit.fiber,
            Visit.file,
            #Visit.prefix,
            Visit.obj,
            Visit.pk.alias("visit_pk"),
            Visit.dateobs.alias("date_obs"),
            Visit.jd,
            Visit.exptime,
            Visit.nframes.alias("n_frames"),
            Visit.assigned,
            Visit.on_target,
            Visit.valid,
            Visit.starflag.alias("spectrum_flags"),
            Visit.catalogid,
            Visit.ra.alias("input_ra"),
            Visit.dec.alias("input_dec"),

            # Source information,
            Visit.gaiadr2_sourceid.alias("gaia_dr2_source_id"),
            CatalogToGaia_DR3.target_id.alias("gaia_dr3_source_id"),
            #Catalog.catalogid.alias("catalogid"),
            Catalog.version_id.alias("version_id"),
            Catalog.lead,
            Catalog.ra,
            Catalog.dec,
            SDSS_ID_Flat.sdss_id,
            SDSS_ID_Flat.n_associated,
            SDSS_ID_Stacked.catalogid21,
            SDSS_ID_Stacked.catalogid25,
            SDSS_ID_Stacked.catalogid31,
            Visit.jmag.alias("j_mag"),
            Visit.jerr.alias("e_j_mag"),            
            Visit.hmag.alias("h_mag"),
            Visit.herr.alias("e_h_mag"),
            Visit.kmag.alias("k_mag"),
            Visit.kerr.alias("e_k_mag"),
            
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
            sq.c.star_pk.alias("star_pk")
        )
        .join(sq, on=(Visit.pk == sq.c.visit_pk)) #
        .switch(Visit)
        # Need to join by Catalog on the visit catalogid (not gaia DR2) because sometimes Gaia DR2 value is 0
        # Doing it like this means we might end up with some `catalogid` actually NOT being v1, but
        # we will have to fix that afterwards. It will be indicated by the `version_id`.
        .join(Catalog, JOIN.LEFT_OUTER, on=(Catalog.catalogid == Visit.catalogid))
        .switch(Visit)
        .join(CatalogToGaia_DR2, JOIN.LEFT_OUTER, on=(Visit.gaiadr2_sourceid == CatalogToGaia_DR2.target_id))
        .join(CatalogToGaia_DR3, JOIN.LEFT_OUTER, on=(CatalogToGaia_DR2.catalogid == CatalogToGaia_DR3.catalogid))
        .switch(Catalog)
        .join(SDSS_ID_Flat, JOIN.LEFT_OUTER, on=(Catalog.catalogid == SDSS_ID_Flat.catalogid))
        .join(SDSS_ID_Stacked, JOIN.LEFT_OUTER, on=(SDSS_ID_Stacked.sdss_id == SDSS_ID_Flat.sdss_id))
        .dicts()
    )


    # The query above will return the same ApogeeVisit when it is associated with multiple sdss_id values,
    # but the .on_conflict_ignore() when upserting will mean that the spectra are not duplicated in the database.
    source_only_keys = (
        "sdss_id",
        "catalogid21",
        "catalogid25",
        "catalogid31",
        "n_associated",
        "catalogid",
        "gaia_dr2_source_id",
        "gaia_dr3_source_id",
        "version_id",
        "lead",
        "ra",
        "dec",
        "j_mag",
        "e_j_mag",
        "h_mag",
        "e_h_mag",
        "k_mag",
        "e_k_mag",
    )
    

    source_data, spectrum_data, matched_sdss_ids = (OrderedDict(), [], {})
    for row in tqdm(q.iterator(), total=limit or 1, desc="Retrieving spectra"):
        basename = row.pop("file")

        catalogid = row["catalogid"]        
        
        this_source_data = dict(zip(source_only_keys, [row.pop(k) for k in source_only_keys]))

        if catalogid in source_data:
            # make sure the only difference is SDSS_ID
            matched_sdss_ids[catalogid] = min(this_source_data["sdss_id"], source_data[catalogid]["sdss_id"])
        else:
            source_data[catalogid] = this_source_data
            matched_sdss_ids[catalogid] = this_source_data["sdss_id"]
        
        row["plate"] = row["plate"].lstrip()
        
        spectrum_data.append({
            "catalogid": catalogid,
            "release": "sdss5",
            "apred": apred,
            "prefix": basename.lstrip()[:2],
            **row
        })

    q_without_rvs = (
        Visit.select(
            Visit.apred,
            Visit.mjd,
            Visit.plate,
            Visit.telescope,
            Visit.field,
            Visit.fiber,
            Visit.file,
            #Visit.prefix,
            Visit.obj,
            Visit.pk.alias("visit_pk"),
            Visit.dateobs.alias("date_obs"),
            Visit.jd,
            Visit.exptime,
            Visit.nframes.alias("n_frames"),
            Visit.assigned,
            Visit.on_target,
            Visit.valid,
            Visit.starflag.alias("spectrum_flags"),
            Visit.catalogid,
            Visit.ra.alias("input_ra"),
            Visit.dec.alias("input_dec"),

            # Source information,
            Visit.gaiadr2_sourceid.alias("gaia_dr2_source_id"),
            CatalogToGaia_DR3.target_id.alias("gaia_dr3_source_id"),
            #Catalog.catalogid.alias("catalogid"),
            Catalog.version_id.alias("version_id"),
            Catalog.lead,
            Catalog.ra,
            Catalog.dec,
            SDSS_ID_Flat.sdss_id,
            SDSS_ID_Flat.n_associated,
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
        .join(RvVisit, JOIN.LEFT_OUTER, on=(Visit.pk == RvVisit.visit_pk))
        .switch(Visit)
        # Need to join by Catalog on the visit catalogid (not gaia DR2) because sometimes Gaia DR2 value is 0
        # Doing it like this means we might end up with some `catalogid` actually NOT being v1, but
        # we will have to fix that afterwards. It will be indicated by the `version_id`.
        .join(Catalog, JOIN.LEFT_OUTER, on=(Catalog.catalogid == Visit.catalogid))
        .switch(Visit)
        .join(CatalogToGaia_DR2, JOIN.LEFT_OUTER, on=(Visit.gaiadr2_sourceid == CatalogToGaia_DR2.target_id))
        .join(CatalogToGaia_DR3, JOIN.LEFT_OUTER, on=(CatalogToGaia_DR2.catalogid == CatalogToGaia_DR3.catalogid))
        .switch(Catalog)
        .join(SDSS_ID_Flat, JOIN.LEFT_OUTER, on=(Catalog.catalogid == SDSS_ID_Flat.catalogid))
        .join(SDSS_ID_Stacked, JOIN.LEFT_OUTER, on=(SDSS_ID_Stacked.sdss_id == SDSS_ID_Flat.sdss_id))
        .where(RvVisit.pk.is_null() & (Visit.apred_vers == apred) & (Visit.catalogid > 0))
        #.where(Visit.pk == 9025065)
        .dicts()
    )


    for row in tqdm(q_without_rvs.iterator(), total=limit or 1, desc="Retrieving bad spectra"):
        basename = row.pop("file")

        assert row["catalogid"] is not None
        catalogid = row["catalogid"]        
        
        this_source_data = dict(zip(source_only_keys, [row.pop(k) for k in source_only_keys]))

        if catalogid in source_data:
            # make sure the only difference is SDSS_ID
            if this_source_data["sdss_id"] is None or source_data[catalogid]["sdss_id"] is None:
                matched_sdss_ids[catalogid] = (this_source_data["sdss_id"] or source_data[catalogid]["sdss_id"])
            else:    
                matched_sdss_ids[catalogid] = min(this_source_data["sdss_id"], source_data[catalogid]["sdss_id"])
        else:
            source_data[catalogid] = this_source_data
            matched_sdss_ids[catalogid] = this_source_data["sdss_id"]
        
        row["plate"] = row["plate"].lstrip()
        
        spectrum_data.append({
            "catalogid": catalogid, # Will be removed later, just for matching sources.
            "release": "sdss5",
            "apred": apred,
            "prefix": basename.lstrip()[:2],
            **row
        })


    # Upsert the sources
    with database.atomic():
        with tqdm(desc="Upserting sources", total=len(source_data)) as pb:
            for chunk in chunked(source_data.values(), batch_size):
                (
                    Source
                    .insert_many(chunk)
                    .on_conflict_ignore()
                    .execute()
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()

    log.info(f"Getting data for sources")
    q = (
        Source
        .select(
            Source.pk,
            Source.sdss_id,
            Source.catalogid,
            Source.catalogid21,
            Source.catalogid25,
            Source.catalogid31
        )
        .tuples()
        .iterator()
    )

    source_pk_by_sdss_id = {}
    source_pk_by_catalogid = {}
    for pk, sdss_id, *catalogids in q:
        if sdss_id is not None:
            source_pk_by_sdss_id[sdss_id] = pk
        for catalogid in catalogids:
            if catalogid is not None:
                source_pk_by_catalogid[catalogid] = pk
        
    for each in spectrum_data:
        catalogid = each["catalogid"]
        try:
            matched_sdss_id = matched_sdss_ids[catalogid]
            source_pk = source_pk_by_sdss_id[matched_sdss_id]            
        except:
            source_pk = source_pk_by_catalogid[catalogid]
            log.warning(f"No SDSS_ID found for catalogid={catalogid}, assigned to source_pk={source_pk}: {each}")

        each["source_pk"] = source_pk
        
    # Upsert the spectra
    pks = upsert_many(
        ApogeeVisitSpectrum,
        ApogeeVisitSpectrum.pk,
        spectrum_data,
        batch_size,
        desc="Upserting spectra"
    )

    # Assign spectrum_pk values to any spectra missing it.
    N = len(pks)
    if pks:
        with tqdm(total=N, desc="Assigning primary keys to spectra") as pb:
            N_assigned = 0
            for batch in chunked(pks, batch_size):
                B =  (
                    ApogeeVisitSpectrum
                    .update(
                        spectrum_pk=Case(None, (
                            (ApogeeVisitSpectrum.pk == pk, spectrum_pk) for spectrum_pk, pk in enumerate_new_spectrum_pks(batch)
                        ))
                    )
                    .where(ApogeeVisitSpectrum.pk.in_(batch))
                    .execute()
                )
                pb.update(B)
                N_assigned += B

        log.info(f"There were {N} spectra inserted and we assigned {N_assigned} spectra with new spectrum_pk values")

    # Sanity check
    q = flatten(
        ApogeeVisitSpectrum
        .select(ApogeeVisitSpectrum.pk)
        .where(ApogeeVisitSpectrum.spectrum_pk.is_null())
        .tuples()
    )
    if q:
        N_updated = 0
        for batch in chunked(q, batch_size):
            N_updated += (
                ApogeeVisitSpectrum
                .update(
                    spectrum_pk=Case(None, [
                        (ApogeeVisitSpectrum.pk == pk, spectrum_pk) for spectrum_pk, pk in enumerate_new_spectrum_pks(batch)
                    ])
                )
                .where(ApogeeVisitSpectrum.pk.in_(batch))
                .execute()            
            )
        log.warning(f"Assigned spectrum_pks to {N_updated} existing spectra")

    assert not (
        ApogeeVisitSpectrum
        .select(ApogeeVisitSpectrum.pk)
        .where(ApogeeVisitSpectrum.spectrum_pk.is_null())
        .exists()
    )


    # Logic: 
    # query TwoMASSPSC based on designation, then to catalog, then everything from there.
    # TODO: This is slow because we are doing one-by-one. consider refactor
    #fix_apvisit_instances_of_invalid_gaia_dr3_source_id()
    log.info(f"Ingested {N} spectra")
    
    
    return N
        

def _migrate_apvisit_metadata(apVisits, raise_exceptions=False):
    def float_or_nan(x):
        try:
            return float(x)
        except:
            return np.nan
    keys_dtypes = {
        "NAXIS1": int,
        "SNR": float_or_nan,
        "NCOMBINE": int, 
        "EXPTIME": float_or_nan,
    }
    K = len(keys_dtypes)
    keys_str = "|".join([f"({k})" for k in keys_dtypes.keys()])

    # 80 chars per line, 150 lines -> 12000
    command_template = " | ".join([
        'hexdump -n 12000 -e \'80/1 "%_p" "\\n"\' {path}',
        f'egrep "{keys_str}"',
        f"head -n {K}"
    ])
    commands = ""
    for apVisit in apVisits:
        path = expand_path(apVisit.path)
        commands += f"{command_template.format(path=path)}\n"
    
    outputs = subprocess.check_output(commands, shell=True, text=True)
    outputs = outputs.strip().split("\n")

    index, all_metadata = (0, {})
    for line in outputs:
        all_metadata.setdefault(apVisits[index].spectrum_pk, {})
        key, value = line.split("=")
        key, value = (key.strip(), value.split()[0].strip(" '"))
        value = keys_dtypes[key](value)
        if key == "NAXIS1":
            # @Nidever: "if there’s 2048 then it hasn’t been dithered, if it’s 4096 then it’s dithered."
            all_metadata[apVisits[index].spectrum_pk]["dithered"] = (value == 4096)
            index += 1            
        elif key == "NCOMBINE":
            all_metadata[apVisits[index].spectrum_pk]["n_frames"] = value
        else:
            all_metadata[apVisits[index].spectrum_pk][key.lower()] = value
            
        
    '''
    if len(outputs) != (K * len(apVisits)):

        if raise_exceptions:
            raise OSError(f"Unexpected outputs from `hexdump` on {apVisits}")

        log.warning(f"Unexpected length of outputs from `hexdump`!")
        log.warning(f"Running this chunk one-by-one to be sure... this chunk goes from {apVisits[0]} to {apVisits[-1]}")
        
        # Do it manually
        all_metadata = {}
        for apVisit in apVisits:
            try:
                this_metadata = _migrate_apvisit_metadata([apVisit], raise_exceptions=True)
            except OSError:
                log.exception(f"Exception on {apVisit}:")
                # Failure mode values.
                all_metadata[apVisit.spectrum_pk] = (False, -1, -1, None)
                if raise_exceptions:
                    raise
                continue
            else:
                all_metadata.update(this_metadata)    
        
        log.info(f"Finished chunk that goes from {apVisits[0]} to {apVisits[-1]} one-by-one")

    else:
        all_metadata = {}
        for apVisit, output in zip(apVisits, chunked(outputs, K)):
            metadata = {}
            for line in output:
                key, value = line.split("=")
                key, value = (key.strip(), value.split()[0].strip(" '"))
                if key in metadata:
                    log.warning(f"Multiple key `{key}` found in {apVisit}: {expand_path(apVisit.path)}")
                    raise a
                else:
                    metadata[key] = value

            # @Nidever: "if there’s 2048 then it hasn’t been dithered, if it’s 4096 then it’s dithered."
            dithered = int(metadata["NAXIS1"]) == 4096
            snr = float(metadata["SNR"])
            n_frames = int(metadata["NCOMBINE"])
            exptime = float(metadata["EXPTIME"])

            all_metadata[apVisit.spectrum_pk] = (dithered, snr, n_frames, exptime)
    '''
    
    return all_metadata



def migrate_apvisit_metadata_from_image_headers(
    where=(ApogeeVisitSpectrum.dithered.is_null() | ApogeeVisitSpectrum.snr.is_null() | ApogeeVisitSpectrum.exptime.is_null()), 
    max_workers: Optional[int] = 8, 
    batch_size: Optional[int] = 100, 
    limit: Optional[int] = None
):
    """
    Gather metadata information from the headers of apVisit files and put that information in to the database.
    
    The header keys it looks for include:
        - `SNR`: the estimated signal-to-noise ratio goes to the `ApogeeVisitSpectrum.snr` attribute
        - `NAXIS1`: for determining `ApogeeVisitSpectrum.dithered` status
        - `NCOMBINE`: for determining the number of frames combined (`ApogeeVisitSpectrum.n_frames`)
        
    :param where: [optional]
        A `where` clause for the `ApogeeVisitSpectrum.select()` statement.
    
    :param max_workers: [optional]
        Maximum number of parallel workers to use.
        
    :param batch_size: [optional]
        The batch size to use when updating `ApogeeVisitSpectrum` objects, and for chunking to workers.

    :param limit: [optional]
        Limit the number of apVisit files to query.
    """

    q = (
        ApogeeVisitSpectrum
        .select()
        .where(where)
        .limit(limit)
        .iterator()
    )

    executor = concurrent.futures.ProcessPoolExecutor(max_workers)

    apVisits, futures, total = ({}, [], 0)
    with tqdm(total=limit or 0, desc="Retrieving metadata", unit="spectra") as pb:
        for chunk in chunked(q, batch_size):
            futures.append(executor.submit(_migrate_apvisit_metadata, chunk))
            for total, apVisit in enumerate(chunk, start=1 + total):
                apVisits[apVisit.spectrum_pk] = apVisit
                pb.update()


    with tqdm(total=total, desc="Collecting results", unit="spectra") as pb:
        for future in concurrent.futures.as_completed(futures):
            for spectrum_pk, meta in future.result().items():
                for key, value in meta.items():
                    setattr(apVisits[spectrum_pk], key, value)
                
                pb.update()

    with tqdm(total=total, desc="Updating", unit="spectra") as pb:     
        for chunk in chunked(apVisits.values(), batch_size):
            pb.update(
                ApogeeVisitSpectrum  
                .bulk_update(
                    chunk,
                    fields=[
                        ApogeeVisitSpectrum.dithered,
                        ApogeeVisitSpectrum.snr,
                        ApogeeVisitSpectrum.n_frames,
                        ApogeeVisitSpectrum.exptime
                    ]
                )
            )

    return pb.n




def migrate_apstar_from_sdss5_database(apred, where=None, limit=None, batch_size=100, max_workers=8):

    from astra.migrations.sdss5db.apogee_drpdb import Star, Visit, RvVisit
    from astra.migrations.sdss5db.catalogdb import CatalogdbModel

    class SDSS_ID_Flat(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_flat"
            
    class SDSS_ID_Stacked(CatalogdbModel):
        class Meta:
            table_name = "sdss_id_stacked"

    # Get SDSS identifiers
    q = (
        Source
        .select(
            Source.pk,
            Source.catalogid,
            Source.catalogid21,
            Source.catalogid25,
            Source.catalogid31
        )        
    )
    source_pks_by_catalogid = {}
    for pk, *catalogids in q.tuples().iterator():
        for catalogid in catalogids:
            if catalogid is not None and catalogid > 0:
                source_pks_by_catalogid[catalogid] = pk
            

    q = (
        Star
        .select(
            Star.obj,
            Star.telescope,
            Star.healpix,
            Star.mjdbeg.alias("min_mjd"),
            Star.mjdend.alias("max_mjd"),
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
            # TODO: doppler_flags? 
            Star.n_components,
            Star.rv_ccpfwhm.alias("ccfwhm"),
            Star.rv_autofwhm.alias("autofwhm"),
            Star.pk.alias("star_pk"),
            Star.catalogid,
            #SDSS_ID_Flat.sdss_id,
        )
        #.join(SDSS_ID_Flat, on=(Star.catalogid == SDSS_ID_Flat.catalogid))
        .where(Star.apred_vers == apred)
    )
    if where is not None:
        q = q.where(where)
    if limit is not None:
        q = q.limit(limit)

    # q = q.order_by(SDSS_ID_Flat.sdss_id.asc())
    
    spectrum_data = {}
    unknown_stars = []
    for star in tqdm(q.dicts().iterator(), total=1, desc="Getting spectra"):
        if star["star_pk"] in spectrum_data:
            raise a
            continue

        star.update(
            release="sdss5",
            filetype="apStar",
            apstar="stars",
            apred=apred,
        )
        # This part assumes that we have already ingested everything from the visits, but that might not be true (e.g., when testing things).
        # TODO: create sources if we dont have them
        #sdss_id = star.pop("sdss_id")
        
        # Sometimes we can get here and there is a source catalogid that we have NEVER seen before, even if we have ingested
        # ALL the visits. The rason is because there are "no good visits". That's fucking annoying, because even if they aren't
        # "good visits", they should appear in the visit table.
        catalogid = star["catalogid"]
        try:
            star["source_pk"] = source_pks_by_catalogid[catalogid]
        except KeyError:
            unknown_stars.append(star)
        else:        
            star.pop("catalogid")
            spectrum_data[star["star_pk"]] = star

    if len(unknown_stars) > 0:
        log.warning(f"There were {len(unknown_stars)} unknown stars")
        for star in unknown_stars:
            if star["n_good_visits"] != 0:
                log.warning(f"Star {star['obj']} (catalogid={star['catalogid']}; star_pk={star['star_pk']} has {star['n_good_visits']} good visits but we've never seen it before")

    # Upsert the spectra
    pks = upsert_many(
        ApogeeCoaddedSpectrumInApStar,
        ApogeeCoaddedSpectrumInApStar.pk,
        list(spectrum_data.values()),
        batch_size,
        desc="Upserting spectra"
    )

    # Assign spectrum_pk values to any spectra missing it.
    N = len(pks)
    if pks:
        with tqdm(total=N, desc="Assigning primary keys to spectra") as pb:
            N_assigned = 0
            for batch in chunked(pks, batch_size):
                B = (
                    ApogeeCoaddedSpectrumInApStar
                    .update(
                        spectrum_pk=Case(None, (
                            (ApogeeCoaddedSpectrumInApStar.pk == pk, spectrum_pk) for spectrum_pk, pk in enumerate_new_spectrum_pks(batch)
                        ))
                    )
                    .where(ApogeeCoaddedSpectrumInApStar.pk.in_(batch))
                    .execute()
                )
                pb.update(B)
                N_assigned += B

        log.info(f"There were {N} spectra inserted and we assigned {N_assigned} spectra with new spectrum_pk values")
    else:
        log.info(f"No new spectra inserted")

    # Do the ApogeeVisitSpectrumInApStar level things based on the FITS files themselves.
    # This is because the APOGEE DRP makes decisions about 'what gets in' to the apStar files, and this information cannot be inferred from the database (according to Nidever).

    executor = concurrent.futures.ProcessPoolExecutor(max_workers)
    q = (
        ApogeeCoaddedSpectrumInApStar
        .select()
        .where(ApogeeCoaddedSpectrumInApStar.release == "sdss5")
        .iterator()
    )

    spectra, futures, total = ({}, [], 0)
    with tqdm(total=limit or 0, desc="Retrieving metadata", unit="spectra") as pb:
        for spectrum in q:
            futures.append(executor.submit(_get_apstar_metadata, spectrum))
            spectra[spectrum.spectrum_pk] = spectrum
            pb.update()

    visit_spectrum_data = []
    failed_spectrum_pks = []
    with tqdm(total=total, desc="Collecting results", unit="spectra") as pb:
        for future in concurrent.futures.as_completed(futures):
            result = future.result()         
            for spectrum_pk, metadata in result.items():
                    
                if metadata is None:
                    failed_spectrum_pks.append(spectrum_pk)
                    continue
                                
                spectrum = spectra[spectrum_pk]

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
                spectrum.v_rad = float(metadata["VRAD"])
                spectrum.e_v_rad = float(metadata["VERR"])
                spectrum.std_v_rad = float(metadata["VSCATTER"])
                spectrum.median_e_v_rad = float(metadata.get("VERR_MED", np.nan))
                spectrum.spectrum_flags = metadata["STARFLAG"]
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
                
                pb.update()


    with tqdm(total=total, desc="Updating", unit="spectra") as pb:     
        for chunk in chunked(spectra.values(), batch_size):
            pb.update(
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
            )

    log.info(f"Creating visit spectra")
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
        .tuples()
    )
    drp_spectrum_data = {}
    for obj, spectrum_pk, telescope, plate, mjd, fiber in tqdm(q.iterator(), desc="Getting DRP spectrum data"):
        drp_spectrum_data.setdefault(obj, {})
        key = "_".join(map(str, (telescope, plate, mjd, fiber)))
        drp_spectrum_data[obj][key] = spectrum_pk

    log.info(f"Matching to DRP spectra")

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
        
    if len(failed_to_match_to_drp_spectrum_pk) > 0:
        log.warning(f"There were {len(failed_to_match_to_drp_spectrum_pk)} spectra that we could not match to DRP spectra")
        log.warning(f"Example: {failed_to_match_to_drp_spectrum_pk[0]}")

    with database.atomic():
        with tqdm(desc="Upserting visit spectra", total=len(only_ingest_visits)) as pb:
            for chunk in chunked(only_ingest_visits, batch_size):
                (
                    ApogeeVisitSpectrumInApStar
                    .insert_many(chunk)
                    .on_conflict_ignore()
                    .execute()
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()

    return N