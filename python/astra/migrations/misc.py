import numpy as np
from astropy.time import Time
import astropy.coordinates as coord
import astropy.units as u
from scipy.signal import argrelmin
from tqdm import tqdm
from peewee import chunked
from peewee import fn, JOIN
import concurrent.futures

from astra.utils import log, flatten
from astra.models.source import Source
from astra.models.apogee import ApogeeVisitSpectrum
from astra.models.boss import BossVisitSpectrum



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



def update_visit_spectra_counts(batch_size=10_000):
    
    # TODO: Switch this to distinct on (mjd, telescope, fiber) etc if you are including multiple reductions
    q_apogee_counts = (
        Source
        .select(
            Source.pk,
            fn.count(ApogeeVisitSpectrum.pk).alias("n_apogee_visits"),
            fn.min(ApogeeVisitSpectrum.mjd).alias("apogee_min_mjd"),
            fn.max(ApogeeVisitSpectrum.mjd).alias("apogee_max_mjd"),
        )
        .join(ApogeeVisitSpectrum, JOIN.LEFT_OUTER, on=(Source.pk == ApogeeVisitSpectrum.source_pk))
        .group_by(Source.pk)
        .dicts()
    )
    q_boss_counts = (
        Source
        .select(
            Source.pk,
            fn.count(BossVisitSpectrum.pk).alias("n_boss_visits"),
            fn.min(BossVisitSpectrum.mjd).alias("boss_min_mjd"),
            fn.max(BossVisitSpectrum.mjd).alias("boss_max_mjd"),            
        )
        .join(BossVisitSpectrum, JOIN.LEFT_OUTER, on=(Source.pk == BossVisitSpectrum.source_pk))
        .group_by(Source.pk)
        .dicts()
    )
    
    # merge counts
    all_counts = {}
    for each in q_apogee_counts:
        all_counts[each.pop("pk")] = each
    
    for each in q_boss_counts:
        all_counts[each.pop("pk")].update(each)
        
    q = { s.pk: s for s in Source.select() }
    
    update = []
    for pk, counts in all_counts.items():
        s = q[pk]
        for k, v in counts.items():
            setattr(s, k, v)
            
        update.append(s)
    
    with tqdm(total=len(update)) as pb:
        for batch in chunked(update, batch_size):
            Source.bulk_update(
                batch,
                fields=[
                    Source.n_apogee_visits,
                    Source.apogee_min_mjd,
                    Source.apogee_max_mjd,
                    Source.n_boss_visits,
                    Source.boss_min_mjd,
                    Source.boss_max_mjd,
                ]
            )
            pb.update(batch_size)
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
    limit=None
):
    #"Sources within 3\" and G_MAG < G_MAG_source + 5"
    
    from astra.migrations.sdss5db.catalogdb import Gaia_DR3

    q = (
        Source
        .select()
        .where(where)
        .limit(limit)
    )

    n_updated = 0
    with tqdm(q) as pb:
        
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
            Source.bulk_update(batch_update, fields=[Source.n_neighborhood])
            pb.update(batch_size)
            
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
        limit=None, batch_size=1000, n_time=256, max_workers=128):
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

    return _compute_f_night_time_for_visits(q, BossVisitSpectrum, get_obs_time, batch_size, n_time, max_workers)

    
def compute_f_night_time_for_apogee_visits(where=ApogeeVisitSpectrum.f_night_time.is_null(), limit=None, batch_size=1000, n_time=256, max_workers=128):
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
    return _compute_f_night_time_for_visits(q, ApogeeVisitSpectrum, lambda v: v.date_obs, batch_size, n_time, max_workers)


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



def _compute_f_night_time_for_visits(q, model, get_obs_time, batch_size, n_time, max_workers):
        
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

    futures, visit_by_pk, observatories = ([], {}, {})
    for visit in tqdm(q.iterator(), desc="Submitting jobs", total=1):
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

    with tqdm(total=len(futures), desc="Computing") as pb:
        for future in concurrent.futures.as_completed(futures):
            pk, f_night_time = future.result()
            visit = visit_by_pk[pk]
            visit.f_night_time = f_night_time
            if not (0 <= f_night_time <= 1):
                log.warning(f"Bad f_night_time for {visit} (f_night_time={f_night_time})")
            updated.append(visit)
            pb.update()                        

            if len(updated) >= batch_size:
                n_updated += (
                    model
                    .bulk_update(
                        updated,
                        fields=[model.f_night_time],
                    )
                )
                updated = []

    if len(updated) > 0:
        n_updated += (
            model
            .bulk_update(
                updated,
                fields=[model.f_night_time],
            )
        )    
        
    return n_updated    
