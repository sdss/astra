import numpy as np
from astropy.time import Time
import astropy.coordinates as coord
import astropy.units as u
from scipy.signal import argrelmin
from peewee import chunked, fn, JOIN
import concurrent.futures

import pickle
from astra.utils import flatten, expand_path
from astra.models.base import database
from astra.models.source import Source
from astra.models.apogee import ApogeeVisitSpectrum
from astra.models.boss import BossVisitSpectrum
from astra.migrations.utils import NoQueue
from astropy.coordinates import SkyCoord
from astropy import units as u


from astra import __version__


von = lambda v: v or np.nan

def compute_w1mag_and_w2mag(
    where=(
        (Source.w1_flux.is_null(False) & Source.w1_mag.is_null(True))
    |   (Source.w2_flux.is_null(False) & Source.w2_mag.is_null(True))
    ),
    limit=None, 
    batch_size=1000,
    queue=None
):
    if queue is None:
        queue = NoQueue()

    q = (
        Source
        .select(
            Source.pk,
            Source.w1_flux,
            Source.w1_dflux,
            Source.w2_flux,
            Source.w2_dflux,
        )
        .where(where)
        .limit(limit)
    )
    n_updated = 0
    queue.put(dict(total=q.count()))

    for batch in chunked(q.iterator(), batch_size):
        
        for source in batch:
            # See https://catalog.unwise.me/catalogs.html (Flux Scale) for justification of 32 mmag offset in W2, and 4 mmag offset in W1
            source.w1_mag = -2.5 * np.log10(von(source.w1_flux)) + 22.5 - 4 * 1e-3 # Vega
            source.e_w1_mag = (2.5 / np.log(10)) * von(source.w1_dflux) / von(source.w1_flux)
            source.w2_mag = -2.5 * np.log10(von(source.w2_flux)) + 22.5 - 32 * 1e-3 # Vega
            source.e_w2_mag = (2.5 / np.log(10)) * von(source.w2_dflux) / von(source.w2_flux)
        
        n_updated += Source.bulk_update(
            batch,
            fields=[
                Source.w1_mag,
                Source.e_w1_mag,
                Source.w2_mag,
                Source.e_w2_mag
            ]
        )
        queue.put(dict(advance=batch_size))

    queue.put(Ellipsis)
    
    return n_updated                



def update_galactic_coordinates(
    where=(Source.ra.is_null(False) & Source.l.is_null(True)),
    limit=None,
    frame="icrs", 
    batch_size=1000,
    queue=None
):
    if queue is None:
        queue = NoQueue()
    
    q = (
        Source
        .select(
            Source.pk,
            Source.ra,
            Source.dec
        )
        .where(where)
        .limit(limit)
    )
    
    n_updated = 0
    queue.put(dict(total=q.count()))
    for batch in chunked(q, batch_size):
        
        coord = SkyCoord(
            ra=[s.ra for s in batch] * u.degree,
            dec=[s.dec for s in batch] * u.degree,
            frame=frame
        )
        
        for source, position in zip(batch, coord.galactic):
            source.l = position.l.value
            source.b = position.b.value
            
        n_updated += Source.bulk_update(
            batch,
            fields=[
                Source.l,
                Source.b
            ]
        )
        queue.put(dict(advance=batch_size))

    queue.put(Ellipsis)    
    return n_updated            


def fix_unsigned_apogee_flags(queue):
    if queue is None:
        queue = NoQueue()

    delta = 2**32 - 2**31
    field_names = [
        "sdss4_apogee_target1_flags",
        "sdss4_apogee_target2_flags",
        "sdss4_apogee2_target1_flags",
        "sdss4_apogee2_target2_flags",
        "sdss4_apogee2_target3_flags",
        "sdss4_apogee_member_flags",
        "sdss4_apogee_extra_target_flags"        
    ]
    updated = {}
    queue.put(dict(total=len(field_names)))
    for field_name in field_names:
        field = getattr(Source, field_name)
        kwds = { field_name: field + delta }
        with database.atomic():        
            updated[field_name] = (
                Source
                .update(**kwds)
                .where(field < 0)
                .execute()
            )
        queue.put(dict(advance=1))
    queue.put(Ellipsis)
    return updated


def compute_gonzalez_hernandez_irfm_effective_temperatures_from_vmk(
    model,
    logg_field,
    fe_h_field,
    where=(
        Source.v_jkc_mag.is_null(False)
    &   Source.k_mag.is_null(False)
    ),
    dwarf_giant_logg_split=3.8,
    batch_size=10_000
):
    '''
    # These are from Table 2 of https://arxiv.org/pdf/0901.3034.pdf
    A_dwarf = np.array([2.3522, -1.8817, 0.6229, -0.0745, 0.0371, -0.0990, -0.0052])
    A_giant = np.array([2.1304, -1.5438, 0.4562, -0.0483, 0.0132, 0.0456, -0.0026])
    
    dwarf_colour_range = [1, 3]
    dwarf_fe_h_range = [-3.5, 0.3]

    giant_colour_range = [0.7, 3.8]
    giant_fe_h_range = [-4.0, 0.1]
    '''
    
    B_dwarf = np.array([0.5201, 0.2511, -0.0118, -0.0186, 0.0408, 0.0033])
    B_giant = np.array([0.5293, 0.2489, -0.0119, -0.0042, 0.0135, 0.0010])
    
    #dwarf_colour_range = [0.1, 0.8] ### WRONG
    dwarf_colour_range = [0.7, 3.0]
    dwarf_fe_h_range = [-3.5, 0.5]
    
    giant_colour_range = [1.1, 3.4]
    giant_fe_h_range = [-4, 0.2]
        
    q = (
        model
        .select(
            model,
            Source,
        )
        .join(Source, on=(model.source_pk == Source.pk), attr="_source")
        .where(
            (model.v_astra == __version__)
        &   logg_field.is_null(False)
        &   fe_h_field.is_null(False)
        )
    )
    
    if where:
        q = q.where(where)

    n_updated, batch = (0, [])
    for row in tqdm(q.iterator()):
        X = (row._source.v_jkc_mag or np.nan) - (row._source.k_mag or np.nan)
        fe_h = getattr(row, fe_h_field.name) or np.nan
        logg = getattr(row, logg_field.name) or np.nan
        
        if logg >= dwarf_giant_logg_split:
            # dwarf
            B = B_dwarf
            valid_v_k = dwarf_colour_range
            valid_fe_h = dwarf_fe_h_range
            row.flag_as_dwarf_for_irfm_teff = True
        else:
            # giant
            B = B_giant
            valid_v_k = giant_colour_range
            valid_fe_h = giant_fe_h_range
            row.flag_as_giant_for_irfm_teff = True
            
        theta = np.sum(B * np.array([1, X, X**2, X*fe_h, fe_h, fe_h**2]))
                
        row.irfm_teff = 5040/theta
        row.flag_out_of_v_k_bounds = not (valid_v_k[0] <= X <= valid_v_k[1])
        row.flag_out_of_fe_h_bounds = not (valid_fe_h[0] <= fe_h <= valid_fe_h[1])
        row.flag_extrapolated_v_mag = (row._source.v_jkc_mag_flag == 0)
        row.flag_poor_quality_k_mag = (
            (row._source.ph_qual is None) 
        or  (row._source.ph_qual[-1] != "A") 
        or  (row._source.e_k_mag > 0.1)
        )
        row.flag_ebv_used_is_upper_limit = row._source.flag_ebv_upper_limit        
        batch.append(row)
        
        if len(batch) >= batch_size:
            model.bulk_update(
                batch,
                fields=[
                    model.irfm_teff,
                    model.irfm_teff_flags,
                ]
            )
            n_updated += batch_size
            batch = []
            
    if len(batch) > 0:
        model.bulk_update(
            batch,
            fields=[
                model.irfm_teff,
                model.irfm_teff_flags,
            ]
        )
        n_updated += len(batch)
    
    return n_updated
        
        

def compute_casagrande_irfm_effective_temperatures(
    model, 
    fe_h_field,
    where=(
        Source.v_jkc_mag.is_null(False)
    &   Source.k_mag.is_null(False)
    ),
    batch_size=10_000
):
    """
    Compute IRFM effective temperatures using the V-Ks colour and the Casagrande et al. (2010) scale.
    """
    
    valid_v_k = [0.78, 3.15]
    
    #https://www.aanda.org/articles/aa/full_html/2010/04/aa13204-09/aa13204-09.html
    a0, a1, a2, a3, a4, a5 = (
        +0.5057,
        +0.2600,
        -0.0146,
        -0.0131,
        +0.0288,
        +0.0016
    )
    
    q = (
        model
        .select(
            model,
            Source,
        )
        .join(Source, on=(model.source_pk == Source.pk), attr="_source")
    )
    
    if where:
        q = q.where(where)

    n_updated, batch = (0, [])
    for row in tqdm(q.iterator()):
        
        X = (row._source.v_jkc_mag or np.nan) - (row._source.k_mag or np.nan)
        fe_h = getattr(row, fe_h_field.name) or np.nan
        theta = a0 + a1 * X + a2*X**2 + a3*X*fe_h + a4*fe_h + a5*fe_h**2
        
        row.irfm_teff = 5040/theta
        row.e_irfm_teff = np.nan
        
        #_source.e_irfm_teff = 5040 * np.sqrt(
        #    (a1 + 2*a2*X + a3*fe_h) ** 2 * _source.e_v_jkc_mag**2
        #+   (a1 + 2*a2*X + a3*fe_h) ** 2 * _source.e_k_mag**2
        #+   (a3*X + a4 + 2*a5*fe_h) ** 2 * _source.e_fe_h**2
        #)
        
        
        row.flag_out_of_v_k_bounds = not (valid_v_k[0] <= X <= valid_v_k[1])
        row.flag_extrapolated_v_mag = (row._source.v_jkc_mag_flag == 0)
        row.flag_poor_quality_k_mag = (row._source.ph_qual is None) or (row._source.ph_qual[-1] != "A")
        row.flag_ebv_used_is_upper_limit = row._source.flag_ebv_upper_limit        
        batch.append(row)
        
        if len(batch) >= batch_size:
            model.bulk_update(
                batch,
                fields=[
                    model.irfm_teff,
                    model.e_irfm_teff,
                    model.irfm_teff_flags,
                ]
            )
            n_updated += batch_size
            batch = []
            
    if len(batch) > 0:
        model.bulk_update(
            batch,
            fields=[
                model.irfm_teff,
                model.e_irfm_teff,
                model.irfm_teff_flags,
            ]
        )
        n_updated += len(batch)
    
    return n_updated



def update_visit_spectra_counts(
    apogee_where=None,
    boss_where=None,
    batch_size=10_000,   
    queue=None 
):
    if queue is None:
        queue = NoQueue()

    sq_apogee = (
        ApogeeVisitSpectrum
        .select(
            ApogeeVisitSpectrum.pk,
            ApogeeVisitSpectrum.source_pk,
            ApogeeVisitSpectrum.telescope, 
            ApogeeVisitSpectrum.mjd, 
            ApogeeVisitSpectrum.fiber,
            ApogeeVisitSpectrum.plate,
            ApogeeVisitSpectrum.field,            
        )
        .distinct(
            ApogeeVisitSpectrum.source_pk,
            ApogeeVisitSpectrum.telescope, 
            ApogeeVisitSpectrum.mjd, 
            ApogeeVisitSpectrum.fiber,
            ApogeeVisitSpectrum.plate,
            ApogeeVisitSpectrum.field,                 
        )    
    )
    if apogee_where is not None:
        sq_apogee = sq_apogee.where(apogee_where)

    q_apogee_counts = (
        ApogeeVisitSpectrum
        .select(
            ApogeeVisitSpectrum.source_pk,
            fn.count(ApogeeVisitSpectrum.pk).alias("n_apogee_visits"),
            fn.min(ApogeeVisitSpectrum.mjd).alias("apogee_min_mjd"),
            fn.max(ApogeeVisitSpectrum.mjd).alias("apogee_max_mjd"),
        )
        .join(sq_apogee, on=(sq_apogee.c.pk == ApogeeVisitSpectrum.pk))
        .group_by(ApogeeVisitSpectrum.source_pk)
        .dicts()
    )

    sq_boss = (
        BossVisitSpectrum
        .select(
            BossVisitSpectrum.pk,
            BossVisitSpectrum.source_pk,
            BossVisitSpectrum.telescope,
            BossVisitSpectrum.mjd,
            BossVisitSpectrum.fieldid,
            BossVisitSpectrum.plateid,
        )
        .distinct(
            BossVisitSpectrum.source_pk,
            BossVisitSpectrum.telescope,
            BossVisitSpectrum.mjd,
            BossVisitSpectrum.fieldid,
            BossVisitSpectrum.plateid,
        )
    )
    if boss_where is not None:
        sq_boss = sq_boss.where(boss_where)
    
    q_boss_counts = (
        BossVisitSpectrum
        .select(
            BossVisitSpectrum.source_pk,
            fn.count(BossVisitSpectrum.pk).alias("n_boss_visits"),
            fn.min(BossVisitSpectrum.mjd).alias("boss_min_mjd"),
            fn.max(BossVisitSpectrum.mjd).alias("boss_max_mjd"),
        )
        .join(sq_boss, on=(sq_boss.c.pk == BossVisitSpectrum.pk))       
        .group_by(BossVisitSpectrum.source_pk)
        .dicts()
    )

    # merge counts
    defaults = dict(
        n_boss_visits=0,
        n_apogee_visits=0,
        apogee_min_mjd=None,
        apogee_max_mjd=None,
        boss_min_mjd=None,
        boss_max_mjd=None
    )
    all_counts = {}
    queue.put(dict(total=q_apogee_counts.count(), description="Querying APOGEE visit counts"))
    for each in q_apogee_counts.iterator():
        source_pk = each.pop("source")
        all_counts[source_pk] = defaults
        all_counts[source_pk].update(each)
        queue.put(dict(advance=1))
    
    queue.put(dict(total=q_boss_counts.count(), description="Querying BOSS visit counts", completed=0))
    for each in q_boss_counts.iterator():
        source_pk = each.pop("source")
        all_counts.setdefault(source_pk, defaults)
        all_counts[source_pk].update(each)
        queue.put(dict(advance=1))
    
    update = []
    queue.put(dict(total=Source.select().count(), description="Collecting source visit counts", completed=0))
    for s in Source.select().iterator():
        for k, v in all_counts.get(s.pk, {}).items():
            setattr(s, k, v)
        update.append(s)

    queue.put(dict(total=len(update), description="Updating source visit counts", completed=0))
    for batch in chunked(update, batch_size):
        # Ugh some issue where if we are only setting Nones for all then if we supply the field it dies
        fields = {"n_apogee_visits", "n_boss_visits"}
        for b in batch:
            if b.apogee_min_mjd is not None:
                fields.add("apogee_min_mjd")
            if b.apogee_max_mjd is not None:
                fields.add("apogee_max_mjd")
            if b.boss_min_mjd is not None:
                fields.add("boss_min_mjd")
            if b.boss_max_mjd is not None:
                fields.add("boss_max_mjd")
        fields = [getattr(Source, f) for f in fields]
        with database.atomic():
            Source.bulk_update(
                batch,
                fields=fields
            )
        queue.put(dict(advance=batch_size))

    queue.put(Ellipsis)
    return len(update)


def compute_n_neighborhood(
    where=(
        (
            Source.n_neighborhood.is_null() 
        |   (Source.n_neighborhood < 0)
        )
        &   Source.gaia_dr3_source_id.is_null(False)    
    ),
    radius=3, # arcseconds
    brightness=5, # magnitudes
    batch_size=1000,
    limit=None,
    queue=None
):
    #"Sources within 3\" and G_MAG < G_MAG_source + 5"
    if queue is None:
        queue = NoQueue()

    from astra.migrations.sdss5db.catalogdb import Gaia_DR3 as _Gaia_DR3
    from sdssdb.peewee.sdss5db import SDSS5dbDatabaseConnection

    class Gaia_DR3(_Gaia_DR3):

        class Meta:
            table_name = 'gaia_dr3_source'
            database = SDSS5dbDatabaseConnection(profile="operations")    

    q = (
        Source
        .select()
        .where(where)
        .limit(limit)
    )

    n_updated = 0
    queue.put(dict(total=limit or q.count()))
    for chunk in chunked(q, batch_size):
        
        batch_sources = {}
        for source in chunk:
            batch_sources[source.gaia_dr3_source_id] = source

        sq = (
            Gaia_DR3
            .select(
                Gaia_DR3.source_id,
                Gaia_DR3.ra,
                Gaia_DR3.dec,
                Gaia_DR3.phot_g_mean_mag
            )
        )
        q_neighbour = (
            Gaia_DR3
            .select(
                Gaia_DR3.source_id,
                fn.count(sq.c.source_id).alias("n_neighborhood")        
            )
            .join(sq, on=(fn.q3c_join(Gaia_DR3.ra, Gaia_DR3.dec, sq.c.ra, sq.c.dec, radius/3600)))
            .where(Gaia_DR3.phot_g_mean_mag > (sq.c.phot_g_mean_mag - brightness))
            .where(Gaia_DR3.source_id.in_(list(batch_sources.keys())))
            .group_by(Gaia_DR3.source_id)
        )
        
        batch_update = []
        for source_id, n_neighborhood in q_neighbour.tuples():
            batch_sources[source_id].n_neighborhood = n_neighborhood - 1 # exclude self
            batch_update.append(batch_sources[source_id])
            
        n_updated += len(batch_update)
        if len(batch_update) > 0:
            Source.bulk_update(batch_update, fields=[Source.n_neighborhood])                
        queue.put(dict(advance=batch_size))

    queue.put(Ellipsis)    
    return n_updated            


def set_missing_gaia_source_ids_to_null():
    (
        Source
        .update(gaia_dr3_source_id=None)
        .where(Source.gaia_dr3_source_id <= 0)
        .execute()
    )
    (
        Source
        .update(gaia_dr2_source_id=None)
        .where(Source.gaia_dr2_source_id <= 0)
        .execute()
    )

def compute_f_night_time_for_boss_visits(
        where=(
            BossVisitSpectrum.f_night_time.is_null()
        &   BossVisitSpectrum.tai_end.is_null(False)
        &   BossVisitSpectrum.tai_beg.is_null(False) # sometimes we don't have tai_beg or tai_end
        ),
        limit=None, batch_size=1000, n_time=256, max_workers=64, queue=None):
    """
    Compute `f_night_time`, which is the observation mid-point expressed as a fraction of time between local sunset and sunrise.
    
    :param where:
        A peewee expression to filter the visits to compute `f_night_time` for.
    
    :param limit:
        The maximum number of visits to compute `f_night_time` for.
    
    :param batch_size:
        The number of visits to update at a time.
    
    :param n_time:
        The number of points to use when computing the sun's position.

    :param max_workers:
        The maximum number of workers to use when computing `f_night_time`.
    """
        
    q = (
        BossVisitSpectrum
        .select()
        .where(where)
        .limit(limit)
    )

    get_obs_time = lambda v: Time((v.tai_beg + 0.5 * (v.tai_end - v.tai_beg))/(24*3600), format="mjd").datetime

    return _compute_f_night_time_for_visits(q, BossVisitSpectrum, get_obs_time, batch_size, n_time, max_workers, queue)

    
def compute_f_night_time_for_apogee_visits(where=ApogeeVisitSpectrum.f_night_time.is_null(), limit=None, batch_size=1000, n_time=256, max_workers=64, queue=None):
    """
    Compute `f_night_time`, which is the observation mid-point expressed as a fraction of time between local sunset and sunrise.
    
    :param where:
        A peewee expression to filter the visits to compute `f_night_time` for.
    
    :param limit:
        The maximum number of visits to compute `f_night_time` for.
    
    :param batch_size:
        The number of visits to update at a time.
    
    :param n_time:
        The number of points to use (per 24 hour period) when computing the sun's position.
    
    :param max_workers:
        The maximum number of workers to use when computing `f_night_time`.        
    """
    q = (
        ApogeeVisitSpectrum
        .select()
        .where(where)
        .limit(limit)
    )
    return _compute_f_night_time_for_visits(q, ApogeeVisitSpectrum, lambda v: v.date_obs, batch_size, n_time, max_workers, queue)


def _compute_f_night_time(pk, observatory, time, n_time):
    # Thanks to Adrian Price-Whelan for blogging about this. This code from his post: https://adrian.pw/blog/sunset-times/
    time_grid = time + np.linspace(-24, 24, 2 * n_time) * u.hour

    sun = coord.get_sun(time_grid[:, None])
    altaz_frame = coord.AltAz(location=observatory, obstime=time_grid[:, None])
    sun_altaz = sun.transform_to(altaz_frame)

    min_idx = np.array(
        [argrelmin(a**2, axis=0, mode="wrap")[0] for a in sun_altaz.alt.degree.T]
    ).flatten()

    # Take the two closest to the middle (e.g., the observing time)
    sunset_idx, sunrise_idx = min_idx[min_idx.searchsorted(n_time) - 1:][:2]

    f_night_time = ((time - time_grid[sunset_idx]) / (time_grid[sunrise_idx] - time_grid[sunset_idx])).value
    return (pk, f_night_time)



def _compute_f_night_time_for_visits(q, model, get_obs_time, batch_size, n_time, max_workers, queue):

    if queue is None:
        queue = NoQueue()
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

    futures, visit_by_pk, observatories = ([], {}, {})
    queue.put(dict(total=q.count()))
    for visit in q.iterator():
        time = Time(get_obs_time(visit))
        observatory_name = visit.telescope[:3].upper()
        try:
            observatory = observatories[observatory_name]
        except:
            observatory = observatories[observatory_name] = coord.EarthLocation.of_site(observatory_name)
        
        futures.append(executor.submit(_compute_f_night_time, visit.pk, observatory, time, n_time))
        visit_by_pk[visit.pk] = visit

    updated = []
    n_updated = 0
    for future in concurrent.futures.as_completed(futures):
        pk, f_night_time = future.result()
        visit = visit_by_pk[pk]
        visit.f_night_time = f_night_time
        updated.append(visit)
        queue.put(dict(advance=1))

        if len(updated) >= batch_size:
            with database.atomic():                    
                n_updated += (
                    model
                    .bulk_update(
                        updated,
                        fields=[model.f_night_time],
                    )
                )
            updated = []

    if len(updated) > 0:
        with database.atomic():
            n_updated += (
                model
                .bulk_update(
                    updated,
                    fields=[model.f_night_time],
                )
            )    
    queue.put(Ellipsis)        
    return n_updated    
